"""Inspect actual downloaded data and metadata, without assuming website counts."""

import argparse
from pathlib import Path

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.utils.io import read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/mermaid")
    p.add_argument("--output", default="outputs/dataset_inventory.json")
    args = p.parse_args()
    d = MermaidDataset(args.root)
    image = read_rgb(d.frames[0].path)
    write_json(
        args.output,
        {
            "images_on_disk": len(d.frames),
            "reference_poses": len(d.poses),
            "images_with_reference_pose": sum(f.reference_c2w is not None for f in d.frames),
            "first_image_shape": list(image.shape),
            "image_bytes": sum(f.path.stat().st_size for f in d.frames),
            "archive_complete": (Path(args.root) / "107179.zip").is_file(),
            "pose_provenance": "Estimated from multi-view bundle adjustment",
            "reference_surface_geometry": False,
            "license": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            "source": "https://doi.org/10.17882/97987",
        },
    )
    print(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
