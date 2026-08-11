"""Fine-tune a conditional ranker for reference-derived triangulation utility."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.geometry.reference import camera_parameters, undistort
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.budget import budget_indices
from underwater_vision.matching.geometry_precision import PrecisionPredictor, augment_coordinates
from underwater_vision.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/conditional_precision_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/observability_ranker_v1"))
    args = parser.parse_args()
    parent = read_json(args.source / "training_v1.json")
    assert parent["status"] == "complete"
    assert (
        read_json("outputs/reference_audit_v1/audit_v1.json")["status"]
        == "passed_proxy_sanity_check"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "executed_source_v1.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PrecisionPredictor(args.source / (parent["selected_architecture"] + "_v1.pt"))
    frames = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    K, distortion = camera_parameters(960, 720, calibration)
    inverse = np.linalg.inv(K)
    records, hashes, angles = {}, {}, []
    for role in ["train", "validation"]:
        records[role] = []
        names = set(parent["training_images" if role == "train" else "validation_images"])
        for path in sorted(args.source.glob(role + "_*.npz")):
            with np.load(path) as z:
                x, error = z["x"].copy(), z["error"].copy()
                a, b = str(z["a"]), str(z["b"])
            assert a in names and b in names
            directions = []
            for xy, name in [(x[:, 4:6], a), (x[:, 6:8], b)]:
                pixels = undistort(xy * [960, 720], K, distortion)
                rays = np.c_[pixels, np.ones(len(pixels))] @ inverse.T
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                directions.append(rays @ frames[name].reference_c2w[:3, :3].T)
            beta = np.arccos(np.clip(np.sum(directions[0] * directions[1], axis=1), -1, 1))
            utility = np.exp(-0.5 * (np.minimum(error, 20) / 0.5) ** 2) * np.minimum(
                1, (np.sin(beta) / np.sin(np.deg2rad(4))) ** 2
            )
            records[role].append((x, utility.astype(np.float32)))
            hashes[path.name] = digest(path)
            angles.extend(np.rad2deg(beta).tolist())
    mean = torch.tensor(predictor.metadata["mean"], device=device)
    std = torch.tensor(predictor.metadata["std"], device=device)
    train = [
        (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        for x, y in records["train"]
    ]
    val = [(torch.from_numpy(x).to(device), y) for x, y in records["validation"]]
    model = predictor.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    best, state, history = -float("inf"), None, []
    for epoch in range(10):
        model.train()
        rng = np.random.default_rng(42 + epoch)
        losses = []
        for i in rng.permutation(len(train)):
            raw, target = train[i]
            x = augment_coordinates(raw, int(rng.integers(4)), bool(rng.integers(2)))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model((x - mean) / std), target
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        retained = []
        with torch.inference_mode():
            for x, y in val:
                score = model((x - mean) / std).cpu().numpy()
                stub = np.zeros((len(score), 3), np.float32)
                stub[:, 2] = 1
                retained.extend(y[budget_indices(stub, score)].tolist())
        objective = float(np.mean(retained))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation_retained_utility": objective,
            }
        )
        if objective > best:
            best = objective
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch{epoch+1}: validation utility {objective:.6f}", flush=True)
    metadata = copy.deepcopy(predictor.metadata)
    metadata.update(
        state_dict=state,
        target="tight_epipolar_precision * min(1,(sin(reference_ray_angle)/sin(4deg))^2)",
        output_interpretation="triangulation_utility_ranking_not_calibrated_probability",
    )
    architecture = metadata["architecture"]
    checkpoint = args.output / f"{architecture}_v1.pt"
    torch.save(metadata, checkpoint)
    write_json(
        args.output / "training_v1.json",
        {
            "status": "complete",
            "selected_architecture": architecture,
            "training_images": parent["training_images"],
            "validation_images": parent["validation_images"],
            "training_candidates": sum(len(x) for x, _ in records["train"]),
            "source_sha256": digest(__file__),
            "parent_training_sha256": digest(args.source / "training_v1.json"),
            "checkpoint_sha256": digest(checkpoint),
            "input_pair_sha256": hashes,
            "history": history,
            "validation_retained_utility": best,
            "angle_quantiles_degrees": np.quantile(angles, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "protocol": "Warm-start validation-selected conditional architecture. Train/validation reference rays only. Fixed4degree observability scale from mapper initialization setting; ten epochs; checkpoint maximizes validation retained utility. Proxy utility is not true3D error. Test regions previously examined; no test labels fit or select this checkpoint.",
        },
    )


if __name__ == "__main__":
    main()
