"""Prepare local 20 validation / 40 test hand-label tasks and 8 repeat tasks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.semantic.protocol import new_run, stratified
from underwater_vision.utils.io import read_rgb, write_json, write_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_labels_v2"))
    p.add_argument("--audit", type=Path, default=Path("outputs/semantic_audit_v2"))
    args = p.parse_args()
    root = new_run(args.output)
    split = json.loads(Path("outputs/reference_audit_v1/spatial_split_v1.json").read_text())
    dataset = MermaidDataset("data/raw/mermaid")
    lookup = {f.name: f for f in dataset.frames}
    entries = []
    (root / "images").mkdir()
    (root / "hand_masks").mkdir()
    (root / "repeat_masks").mkdir()
    for role, count in [("validation", 20), ("test", 40)]:
        names = stratified(split["groups"][role], count)
        # Test tasks sorted by reference spatial coordinate, with 3 descriptive blocks.
        if role == "test":
            spec = split["specification"]
            names.sort(
                key=lambda n: float(
                    (lookup[n].reference_c2w[:3, 3] - spec["origin"]) @ np.asarray(spec["axis"])
                )
            )
        for i, name in enumerate(names):
            rgb, _ = resize_rgb(read_rgb(lookup[name].path), 960)
            image = root / "images" / name
            write_rgb(image, rgb)
            pseudo = args.audit / "teacher" / (Path(name).stem + ".png")
            # Suggestions stay separate; pending mask starts ignore, never 'reviewed'.
            mask = root / "hand_masks" / (Path(name).stem + ".png")
            Image.fromarray(np.full(rgb.shape[:2], 255, np.uint8)).save(mask)
            entries.append(
                {
                    "image_id": name,
                    "image": str(image.resolve()),
                    "mask": str(mask.resolve()),
                    "slab": role,
                    "role": role,
                    "label_source": "hand",
                    "reviewed": False,
                    "annotator": None,
                    "date": None,
                    "pass": None,
                    "mask_sha256": None,
                    "test_block": min(2, i * 3 // count) if role == "test" else None,
                    "suggestion": str(pseudo.resolve()) if pseudo.exists() else None,
                }
            )
    for name in split["groups"]["train"]:
        entries.append(
            {
                "image_id": name,
                "image": str(lookup[name].path.resolve()),
                "mask": str((root / "pseudo_train" / (Path(name).stem + ".png")).resolve()),
                "slab": "train",
                "role": "train",
                "label_source": "pseudo",
                "reviewed": False,
            }
        )
    write_json(
        root / "manifest.json",
        {
            "version": "semantic_v2",
            "frozen": None,
            "spatial_split": split,
            "entries": entries,
            "repeat_tasks": [e["image_id"] for e in entries if e["role"] == "validation"][:8],
            "repeat_minimum_days": 2,
            "note": "Suggestions never count as hand labels. Freeze only after human review and repeat consistency.",
        },
    )
    write_json(
        root / "annotation_protocol.json",
        {
            "classes": [
                "sand_sediment",
                "rock",
                "statue",
                "tile_artifacts",
                "sessile_life",
                "dynamic",
            ],
            "ignore": 255,
            "boundary_rule": "Mark uncertain boundaries and water column ignore; distinguish statue/base from surrounding rock",
            "repeat": "Relabel the eight repeat tasks without viewing pass 1, at least 48 hours later",
            "evaluation": "Test labels are evaluation only, never temperature or threshold selection",
        },
    )
    print(
        f"Prepared {20+40} hand tasks, 8 repeat tasks and {len(split['groups']['train'])} train placeholders at {root}"
    )


if __name__ == "__main__":
    main()
