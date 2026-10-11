"""Export standalone probabilities or teacher suggestions without fitting models."""

import argparse
from pathlib import Path

from PIL import Image

from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.semantic.infer import SemanticPredictor
from underwater_vision.semantic.protocol import load_manifest, new_run
from underwater_vision.semantic.pseudo_labels import GroundedSAMTeacher
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--role", choices=["train", "validation", "test", "hand", "all"], default="hand")
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_suggestions_v2"))
    p.add_argument(
        "--suggestions",
        action="store_true",
        help="Copy proposals into editable hand-label bundle, never into reviewed masks",
    )
    a = p.parse_args()
    if bool(a.checkpoint) == a.teacher:
        p.error("Choose --checkpoint or --teacher")
    manifest = load_manifest(a.manifest)
    if a.suggestions and manifest.get("frozen"):
        raise ValueError("Do not alter a frozen annotation bundle")
    root = new_run(a.output)
    model = GroundedSAMTeacher() if a.teacher else SemanticPredictor(a.checkpoint)
    if a.teacher:
        write_json(root / "teacher_provenance.json", model.provenance)
    for e in manifest["entries"]:
        if a.role != "all" and not (
            e["role"] == a.role or a.role == "hand" and e["role"] in {"validation", "test"}
        ):
            continue
        rgb, _ = resize_rgb(read_rgb(e["image"]), 960)
        stem = Path(e["image_id"]).stem
        if a.teacher:
            result = model.predict(rgb)
            mask = result["mask"]
            atomic_npz(root / (stem + ".npz"), mask=mask, confidence=result["confidence"])
            write_json(root / (stem + ".json"), {"proposals": result["proposals"]})
        else:
            probability = model.cached(e["image"], 960)
            mask = probability.argmax(0).astype("uint8")
            atomic_npz(
                root / (stem + ".npz"),
                probabilities=probability,
                checkpoint_hash=model.checkpoint_hash,
            )
        Image.fromarray(mask).save(root / (stem + ".png"))
        if a.suggestions:
            target = a.manifest.parent / "suggestions" / (stem + ".png")
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise FileExistsError("Suggestion already exists")
            Image.fromarray(mask).save(target)
        print(f"Exported {e['role']} inference: {stem}", flush=True)


if __name__ == "__main__":
    main()
