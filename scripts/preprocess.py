"""Export aspect-preserving development images and an exact scale manifest."""

import argparse
from pathlib import Path

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.utils.io import read_rgb, write_json, write_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/mermaid")
    p.add_argument("--output", default="data/processed/mermaid", type=Path)
    p.add_argument("--max-size", default=960, type=int)
    p.add_argument("--count", default=24, type=int)
    p.add_argument("--stride", default=2, type=int)
    p.add_argument("--enhance", action="store_true")
    args = p.parse_args()
    records = []
    for frame in MermaidDataset(args.root).select(count=args.count, stride=args.stride):
        image, scale = resize_rgb(read_rgb(frame.path), args.max_size, args.enhance)
        write_rgb(args.output / frame.name, image)
        records.append(
            {
                "name": frame.name,
                "source": str(frame.path.resolve()),
                "width": image.shape[1],
                "height": image.shape[0],
                "scale_xy": list(scale),
            }
        )
    write_json(args.output / "manifest.json", records)


if __name__ == "__main__":
    main()
