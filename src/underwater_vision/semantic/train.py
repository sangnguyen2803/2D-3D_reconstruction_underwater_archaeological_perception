"""Segmentation training after human Gate 0 and frozen hand-label protocol."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from underwater_vision.evaluation.segmentation import (
    confusion,
    fit_temperature,
    pixel_ece,
    segmentation_metrics,
)
from underwater_vision.semantic.dataset import SemanticDataset
from underwater_vision.semantic.model import SegmentationModel
from underwater_vision.semantic.protocol import digest_file, load_manifest, new_run, require_gate
from underwater_vision.semantic.schema import IGNORE
from underwater_vision.utils.io import write_json


def masked_loss(logits, target, smoothing=0.05):
    if not (target != IGNORE).any():
        return logits.sum() * 0
    return F.cross_entropy(logits, target, ignore_index=IGNORE, label_smoothing=smoothing)


def validation_outputs(model, loader, device, max_calibration_pixels=200000):
    matrices = []
    logits_list, targets = [], []
    model.eval()
    with torch.no_grad():
        for image, target, _ in loader:
            output = model(image.to(device)).float().cpu().numpy().transpose(0, 2, 3, 1)
            target = target.numpy()
            for logit, mask in zip(output, target, strict=True):
                matrices.append(confusion(mask, logit.argmax(-1).astype(np.uint8)))
                valid = mask != IGNORE
                logits_list.append(logit[valid])
                targets.append(mask[valid])
    x = np.concatenate(logits_list)
    y = np.concatenate(targets)
    if len(y) > max_calibration_pixels:
        indices = np.random.default_rng(42).choice(len(y), max_calibration_pixels, replace=False)
        x, y = x[indices], y[indices]
    return segmentation_metrics(np.sum(matrices, axis=0)), x, y


def train_segmentation(config, output):
    require_gate(config["gate"])
    manifest = load_manifest(config["manifest"], require_frozen=True)
    import yaml

    criteria = yaml.safe_load(Path(config["criteria"]).read_text())
    if not criteria.get("frozen"):
        raise ValueError("Success criteria must be frozen after Gate 0 before training")
    root = new_run(output)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merge_map = yaml.safe_load(Path("configs/semantic/labels.yaml").read_text()).get(
        "merge_map", {}
    )
    train = SemanticDataset(
        config["manifest"], "train", config["size"], augment=True, merge_map=merge_map
    )
    validation = SemanticDataset(
        config["manifest"], "validation", config["size"], merge_map=merge_map
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0
    )
    vl = DataLoader(validation, batch_size=config["batch_size"], num_workers=0)
    specification = yaml.safe_load(Path(config["model_config"]).read_text())
    model = SegmentationModel(specification).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    amp = device.type == "cuda" and config["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -1
    state = None
    history = []
    write_json(root / "resolved_config.json", config)
    write_json(root / "input_manifest.json", manifest)
    accumulation = config["gradient_accumulation"]
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (image, target, _) in enumerate(loader):
            # Correctly average the final short accumulation group too.
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = masked_loss(
                    model(image.to(device)), target.to(device), config["label_smoothing"]
                )
            scaler.scale(loss / group_size).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metric, _, _ = validation_outputs(model, vl, device)
        if metric["miou"] is None:
            raise ValueError("No hand-labeled validation pixels")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_miou": metric["miou"],
            }
        )
        if metric["miou"] > best:
            best = metric["miou"]
            state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        schedule.step()
        print(history[-1], flush=True)
    model.load_state_dict(state)
    metric, x, y = validation_outputs(model, vl, device)
    temperature = fit_temperature(x, y)
    from scipy.special import softmax

    probabilities = softmax(x / temperature, axis=1)[:, 5]
    dynamic = y == 5
    thresholds = np.linspace(0.1, 0.95, 18)
    f1 = [
        2
        * np.sum((probabilities >= t) & dynamic)
        / max(np.sum(probabilities >= t) + dynamic.sum(), 1)
        for t in thresholds
    ]
    dynamic_threshold = float(thresholds[int(np.argmax(f1))]) if dynamic.any() else 1.0
    saved = {
        "state_dict": state,
        "status": "trained",
        "merge_map": merge_map,
        "feature_spec": "segmentation_six_v2",
        "model_config": specification,
        "size": config["size"],
        "temperature": temperature,
        "dynamic_threshold": dynamic_threshold,
        "seed": seed,
        "training_images": [e["image_id"] for e in manifest["entries"] if e["role"] == "train"],
        "validation_images": [
            e["image_id"] for e in manifest["entries"] if e["role"] == "validation"
        ],
        "manifest_sha256": digest_file(config["manifest"]),
        "criteria_sha256": digest_file(config["criteria"]),
        "label_provenance": "pseudo train; human-reviewed validation; test never used",
    }
    if specification["kind"] == "segformer":
        saved["model_config"]["hf_config"] = model.network.config.to_dict()
        saved["model_config"]["hf_config"].pop("num_labels", None)
    checkpoint = root / "segmentation_v2.pt"
    torch.save(saved, checkpoint)
    result = {
        "validation": metric,
        "validation_ece_before": pixel_ece(x, y),
        "validation_ece_after": pixel_ece(x, y, temperature),
        "temperature": temperature,
        "dynamic_threshold": dynamic_threshold,
        "dynamic_validation_pixels": int(dynamic.sum()),
        "history": history,
        "checkpoint_sha256": digest_file(checkpoint),
        "seed": seed,
    }
    write_json(root / "training_v2.json", result)
    if config.get("tracking"):
        import mlflow

        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
        mlflow.set_experiment("semantic_v2")
        with mlflow.start_run(run_name=root.name):
            mlflow.log_params({"seed": seed, "kind": specification["kind"]})
            mlflow.log_metrics({"validation_miou": best, "temperature": temperature})
            mlflow.log_artifacts(str(root))
    return result
