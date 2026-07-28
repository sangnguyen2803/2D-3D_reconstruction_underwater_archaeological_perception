"""Optional YOLO11n dynamic detector, permitted only by reviewed Gate 0 counts."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from underwater_vision.semantic.detection import mask_boxes
from underwater_vision.semantic.protocol import load_manifest, new_run, require_gate
from underwater_vision.utils.io import read_rgb, write_json, write_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--output", type=Path, default=Path("outputs/dynamic_detector_v2"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    gate = require_gate("outputs/semantic_audit_v2/gate_review.json")
    if not str(gate.get("detector", "")).startswith("eligible"):
        raise ValueError(
            "Detector skipped: human Gate 0 did not meet instance/frequency thresholds"
        )
    manifest = load_manifest(a.manifest, require_frozen=True)
    from ultralytics import YOLO

    root = new_run(a.output)
    for e in manifest["entries"]:
        role = e["role"]
        if role not in {"train", "validation", "test"}:
            continue
        mask = np.asarray(Image.open(e["mask"]))
        rgb = read_rgb(e["image"])
        rgb = cv2_resize(rgb, mask.shape)
        target = root / "images" / role / (Path(e["image_id"]).stem + ".jpg")
        write_rgb(target, rgb)
        label = root / "labels" / role / (target.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "\n".join(" ".join(map(str, box)) for box in mask_boxes(mask)), encoding="utf-8"
        )
    dataset = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "dynamic"},
    }
    dataset_path = root / "detector_data_v2.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(dataset_path),
        epochs=a.epochs,
        imgsz=512,
        batch=4,
        seed=a.seed,
        project=str(root),
        name="train_v2",
        exist_ok=False,
    )
    best = YOLO(str(root / "train_v2/weights/best.pt"))
    metrics = best.val(data=str(dataset_path), split="test", project=str(root), name="test_v2")
    write_json(
        root / "evaluation_v2.json",
        {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "seed": a.seed,
            "boxes": "Connected components of dynamic masks; pseudo train, reviewed hand test",
            "limitation": "Instance boundaries are approximated by semantic components; touching objects may merge",
            "license": "Ultralytics AGPL-3.0; separate licensing may be required for different distribution/use",
        },
    )


def cv2_resize(rgb, shape):
    import cv2

    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
