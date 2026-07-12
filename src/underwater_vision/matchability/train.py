"""Seeded sparse-supervision training; spatial validation AP selects weights."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from underwater_vision.evaluation.matchability import metrics
from underwater_vision.matchability.calibration import fit_temperature, probabilities
from underwater_vision.matchability.dataset import MatchabilityDataset
from underwater_vision.matchability.frozen_cache import FrozenEvaluationCache
from underwater_vision.matchability.loss import sample_logits, sparse_bce
from underwater_vision.matchability.model import MatchabilityModel
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.utils.io import atomic_npz, write_json


@torch.inference_mode()
def evaluate_logits(model, dataset, device):
    model.eval()
    rows = []
    if model.kind != "dino_head":
        # Eval BatchNorm is fixed: batching full RGB views amortizes GPU launches.
        for start in range(0, len(dataset), 4):
            items = [dataset[i] for i in range(start, min(start + 4, len(dataset)))]
            rgb = torch.stack([item["rgb"] for item in items]).to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                out = model(rgb)
            for i, item in enumerate(items):
                z = sample_logits(out[i : i + 1].float(), item["xy"].to(device))[0].cpu().numpy()
                rows.append({"name": item["name"], "logits": z, "labels": item["labels"].numpy()})
        return rows
    frozen_cache = FrozenEvaluationCache() if device == "cuda" else None
    for item in dataset:
        with torch.autocast(device, enabled=device == "cuda"):
            if model.kind == "dino_head" and device == "cuda":
                path = dataset.feature_root / "images" / (item["name"] + ".png")
                grid = frozen_cache.grid(model, item["rgb"][None].to(device), path)
                out = model.decode_grid(grid, item["rgb"].shape[-2:])
            else:
                out = model(item["rgb"][None].to(device))
        z = sample_logits(out.float(), item["xy"].to(device))[0].cpu().numpy()
        rows.append({"name": item["name"], "logits": z, "labels": item["labels"].numpy()})
    return rows


def fit(args):
    root = new_run(args.output)
    cfg = yaml.safe_load(Path(args.config).read_text())
    criteria = yaml.safe_load(Path(args.criteria).read_text())
    if criteria.get("status") != "frozen_after_E0":
        raise ValueError("Freeze criteria after E0 before final training")
    manifest = read_json(Path(args.dataset) / "manifest_v3.json")
    split = next(p for p in manifest["protocols"] if p["name"] == args.protocol)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = MatchabilityDataset(
        args.dataset,
        split["groups"]["train"],
        args.protocol,
        args.graph,
        args.variant,
        cfg["crop_hw"],
        args.seed,
    )
    validation = MatchabilityDataset(
        args.dataset, split["groups"]["validation"], args.protocol, args.graph
    )
    test = MatchabilityDataset(
        args.dataset, split["groups"]["test"], args.protocol, args.graph, cache_rgb=False
    )
    model_cfg = yaml.safe_load(Path(args.model).read_text())
    model = MatchabilityModel(model_cfg, pretrained=True).to(device)
    if model.kind == "segformer_b0":
        model_cfg["hf_config"] = model.network.config.to_dict()
        model_cfg["hf_config"].pop("num_labels", None)
        model_cfg["hf_config"].pop("id2label", None)
        model_cfg["hf_config"].pop("label2id", None)
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                encoder
                if name.startswith("encoder.") or name.startswith("network.segformer.")
                else head
            ).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": cfg["encoder_lr"]}, {"params": head, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    n_pos = n_neg = 0
    for name in train.names:
        with np.load(
            Path(args.dataset) / "labels" / args.protocol / args.graph / (name + ".npz")
        ) as z:
            y = z[args.variant]
            known = y >= 0
            n_pos += float(y[known].sum())
            n_neg += float((1 - y[known]).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training labels need both positive and negative supervision")
    pos_weight = n_neg / max(1, n_pos)
    provenance = {
        "model_config": model_cfg,
        "train_config": cfg,
        "epochs": args.epochs,
        "seed": args.seed,
        "protocol": args.protocol,
        "variant": args.variant,
        "graph": args.graph,
        "criteria_sha256": digest(args.criteria),
        "manifest_sha256": digest(Path(args.dataset) / "manifest_v3.json"),
        "label_spec": manifest["label_spec"],
        "label_audit_sha256": digest(Path(args.dataset) / "audit_v3.json"),
        "pos_weight": pos_weight,
        "training_images": train.names,
        "validation_images": validation.names,
        "test_images": test.names,
        "spatial_split": split,
        "evaluation_batch": 1 if model.kind == "dino_head" else 4,
        "frozen_eval_cache": (
            "lossless_fp32_original_strides"
            if model.kind == "dino_head" and device == "cuda"
            else None
        ),
    }
    write_json(root / "resolved_config_v3.json", provenance)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("dense_matchability_v3")
    best, best_state, history = -1.0, None, []
    start = time.perf_counter()
    with mlflow.start_run(run_name=root.name):
        mlflow.log_params(
            {
                "model": model.kind,
                "fold": args.protocol,
                "seed": args.seed,
                "epochs": args.epochs,
                "variant": args.variant,
                "encoder_lr": cfg["encoder_lr"],
                "head_lr": cfg["head_lr"],
                "criteria_sha256": provenance["criteria_sha256"],
            }
        )
        for epoch in range(args.epochs):
            model.train()
            order = np.random.default_rng(args.seed + epoch).permutation(len(train))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            accumulation = cfg["accumulation"]
            for step, idx in enumerate(order):
                item = train[int(idx)]
                with torch.autocast(device, enabled=device == "cuda"):
                    out = model(item["rgb"][None].to(device))
                    loss = sparse_bce(
                        out, item["xy"].to(device), item["labels"].to(device), pos_weight
                    )
                group_size = min(accumulation, len(order) - (step // accumulation) * accumulation)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(order):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            scheduler.step()
            rows = evaluate_logits(model, validation, device)
            z, y = np.concatenate([r["logits"] for r in rows]), np.concatenate(
                [r["labels"] for r in rows]
            )
            known = y >= 0
            ap = float(average_precision_score(y[known], z[known]))
            record = {"epoch": epoch + 1, "train_bce": float(np.mean(losses)), "validation_ap": ap}
            history.append(record)
            mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=epoch + 1)
            print(f"{root.name} epoch {epoch + 1}/{args.epochs} validation AP={ap:.5f}", flush=True)
            if ap > best:
                best = ap
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if model.kind != "dino_head" or not k.startswith("encoder.")
                }
        model.load_state_dict(best_state, strict=model.kind != "dino_head")
        rows = evaluate_logits(model, validation, device)
        z, y = np.concatenate([r["logits"] for r in rows]), np.concatenate(
            [r["labels"] for r in rows]
        )
        temperature = fit_temperature(z, y)
        checkpoint = {
            "status": "trained",
            "feature_spec": "dense_matchability_v3",
            "state_dict": best_state,
            "temperature": temperature,
            **provenance,
        }
        torch.save(checkpoint, root / "model_v3.pt")
        report = {
            "status": "complete",
            "best_validation_ap": best,
            "history": history,
            "temperature": temperature,
            "checkpoint_hash": digest(root / "model_v3.pt"),
            "elapsed_seconds": time.perf_counter() - start,
            "device": device,
            "validation_before": metrics(y, probabilities(z)),
            "validation_after": metrics(y, probabilities(z, temperature)),
            "spatial_regions": 1,
            "selection_rule": "spatial validation AP only; temperature fitted on validation; test never fits anything",
        }
        (root / "predictions").mkdir()
        test_rows = evaluate_logits(model, test, device)
        for row in test_rows:
            atomic_npz(
                root / "predictions" / (row["name"] + ".npz"),
                **row,
                probability=probabilities(row["logits"], temperature),
            )
        tz, ty = np.concatenate([r["logits"] for r in test_rows]), np.concatenate(
            [r["labels"] for r in test_rows]
        )
        report["test_before"], report["test_after"] = metrics(ty, probabilities(tz)), metrics(
            ty, probabilities(tz, temperature)
        )
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(root / "evaluation_v3.json", report)
        mlflow.log_metrics(
            {
                "test_ap": report["test_after"]["average_precision"],
                "test_ece": report["test_after"]["ece"],
                "temperature": temperature,
            }
        )
    return root
