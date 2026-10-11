"""Train geometry precision on contrast0.01 survivors with spatial train/val."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.features.local import extract_sift
from underwater_vision.geometry.reference import camera_parameters, reference_labels
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.budget import budget_indices
from underwater_vision.matching.geometry_precision import (
    FEATURES,
    augment_coordinates,
    precision_features,
)
from underwater_vision.matching.learned_filter import MatchFilter, candidates
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json
from underwater_vision.visibility.deterministic import estimate_visibility, sample_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/conditional_precision_v1"))
    args = parser.parse_args()
    audit = read_json("outputs/reference_audit_v1/audit_v1.json")
    assert audit["status"] == "passed_proxy_sanity_check"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "executed_source_v1.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    dataset_root = Path("outputs/matchability_labels_v3")
    split = next(
        p
        for p in read_json(dataset_root / "manifest_v3.json")["protocols"]
        if p["name"] == "v1_partition"
    )
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    selected = {role: sorted(split["groups"][role]) for role in ["train", "validation"]}
    assert not set(selected["train"]) & set(selected["validation"])
    report = {
        "status": "building",
        "protocol": "Fresh MLP/context networks learn soft tight-epipolar precision only among mutual ratio0.8 matches from contrast0.01 RootSIFT. No DINO feature or SegFormer score enters inputs. All226 spatial train and65 validation views; training offsets1/6 reduce compute while spanning near/far pairs, validation offsets1..6. No test image or reference label fits a model. Ten epochs, joint coordinate-axis augmentation, validation-only retained clipped-error checkpoint/architecture selection. Ranking score, not calibrated match probability.",
        "training_images": selected["train"],
        "validation_images": selected["validation"],
        "source_sha256": digest(__file__),
    }
    write_json(args.output / "protocol_v1.json", report)
    records = {}
    for role, names in selected.items():
        features = {}
        for index, name in enumerate(names):
            rgb = read_rgb(dataset_root / "images" / (name + ".png"))
            xy, local = extract_sift(rgb, 4096, 0.01)
            reliability, _ = estimate_visibility(rgb)
            features[name] = {
                "xy": xy,
                "local": local,
                "quality": sample_map(reliability, xy),
                "semantic": np.zeros((len(xy), 1)),
                "shape": np.array(rgb.shape[:2]),
            }
            if (index + 1) % 50 == 0:
                print(f"{role}: features {index+1}/{len(names)}", flush=True)
        numeric = {int(n.split("_")[-1].split(".")[0]): n for n in names}
        offsets = [1, 6] if role == "train" else range(1, 7)
        pairs = [
            (numeric[i], numeric[i + offset])
            for i in sorted(numeric)
            for offset in offsets
            if i + offset in numeric
        ]
        records[role] = []
        for index, (a, b) in enumerate(pairs):
            fa, fb = features[a], features[b]
            ids, x = candidates(fa, fb, 960, 720)
            keep = (x[:, 1] < 0.8) & (x[:, 2] > 0.5)
            ids = ids[keep]
            if not len(ids):
                continue
            px = precision_features(fa, fb, ids, ratios=x[keep, 1])
            K, distortion = camera_parameters(960, 720, calibration)
            _, error, _ = reference_labels(
                fa["xy"][ids[:, 0]],
                fb["xy"][ids[:, 1]],
                lookup[a].reference_c2w,
                lookup[b].reference_c2w,
                K,
                distortion,
            )
            finite = np.isfinite(error)
            px, error = px[finite], error[finite]
            atomic_npz(
                args.output / f"{role}_{index:04}.npz",
                x=px,
                error=error,
                a=np.asarray(a),
                b=np.asarray(b),
            )
            records[role].append((px, error))
            if index % 25 == 0:
                print(f"{role}: labeled {index+1}/{len(pairs)} pairs", flush=True)
        features.clear()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_x = np.concatenate([r[0] for r in records["train"]])
    mean, std = all_x.mean(0), np.maximum(all_x.std(0), 1e-3)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    train = [
        (
            torch.from_numpy(x).to(device),
            torch.from_numpy(
                np.exp(-0.5 * np.square(np.minimum(e, 20) / 0.5)).astype(np.float32)
            ).to(device),
        )
        for x, e in records["train"]
        if len(x)
    ]
    val = [
        (torch.from_numpy((x - mean) / std).to(device), e)
        for x, e in records["validation"]
        if len(x)
    ]
    outputs = {}
    for architecture in ["mlp", "context"]:
        torch.manual_seed(42)
        model = MatchFilter(architecture, channels=len(FEATURES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
        best, state, history = float("inf"), None, []
        for epoch in range(10):
            model.train()
            rng = np.random.default_rng(42 + epoch)
            losses = []
            for i in rng.permutation(len(train)):
                raw, target = train[i]
                augmented = augment_coordinates(raw, int(rng.integers(4)), bool(rng.integers(2)))
                logits = model((augmented - mean_t) / std_t)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            selected_errors = []
            with torch.inference_mode():
                for x, e in val:
                    score = model(x).cpu().numpy()
                    stub = np.zeros((len(score), 3), np.float32)
                    stub[:, 2] = 1
                    ids = budget_indices(stub, score)
                    selected_errors.extend(np.minimum(e[ids], 3).tolist())
            objective = float(np.mean(selected_errors))
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "validation_retained_clipped_error": objective,
                }
            )
            if objective < best:
                best, state = objective, {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(f"{architecture} epoch{epoch+1}: validation error {objective:.6f}", flush=True)
        metadata = {
            "state_dict": state,
            "architecture": architecture,
            "feature_spec": "conditional_geometry_v1_10",
            "feature_names": FEATURES,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "contrast_threshold": 0.01,
            "max_features": 4096,
            "image_size": 960,
            "training_images": selected["train"],
            "validation_images": selected["validation"],
            "target": "exp(-0.5*(reference_epipolar_error/0.5)^2)",
            "output_interpretation": "conditional_precision_ranking_not_calibrated_probability",
        }
        torch.save(metadata, args.output / f"{architecture}_v1.pt")
        outputs[architecture] = {
            "validation_objective": best,
            "history": history,
            "checkpoint_sha256": digest(args.output / f"{architecture}_v1.pt"),
        }
    winner = min(outputs, key=lambda k: outputs[k]["validation_objective"])
    report.update(
        status="complete",
        training_candidates=len(all_x),
        training_pairs=len(train),
        validation_pairs=len(val),
        models=outputs,
        selected_architecture=winner,
        device=device,
    )
    write_json(args.output / "training_v1.json", report)
    print("Validation-selected architecture:", winner, flush=True)


if __name__ == "__main__":
    main()
