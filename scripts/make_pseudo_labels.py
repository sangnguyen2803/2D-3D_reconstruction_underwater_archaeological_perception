"""Generate train-slab Grounded SAM labels after a passed human Gate 0."""

import argparse
from pathlib import Path

from PIL import Image

from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.semantic.protocol import load_manifest, new_run, require_gate
from underwater_vision.semantic.pseudo_labels import GroundedSAMTeacher
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--gate", type=Path, default=Path("outputs/semantic_audit_v2/gate_review.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/pseudo_labels_v2"))
    a = p.parse_args()
    require_gate(a.gate)
    manifest = load_manifest(a.manifest)
    root = new_run(a.output)
    teacher = GroundedSAMTeacher()
    write_json(root / "teacher_provenance.json", teacher.provenance)
    for i, e in enumerate(e for e in manifest["entries"] if e["role"] == "train"):
        rgb, _ = resize_rgb(read_rgb(e["image"]), 960)
        result = teacher.predict(rgb)
        stem = Path(e["image_id"]).stem
        atomic_npz(root / (stem + ".npz"), mask=result["mask"], confidence=result["confidence"])
        mask_path = Path(e["mask"])
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_path.exists():
            raise FileExistsError("Never replace an existing label mask")
        Image.fromarray(result["mask"]).save(mask_path)
        write_json(root / (stem + ".json"), {"proposals": result["proposals"]})
        print(f"Pseudo train view {i+1}: {stem}", flush=True)


if __name__ == "__main__":
    main()
