"""Freeze an image list so concurrent data downloads cannot change an experiment."""

import argparse
from pathlib import Path

from underwater_vision.data.dataset_loader import MermaidDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/mermaid")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    dataset = MermaidDataset(args.root)
    names = {int(f.path.stem.rsplit("_", 1)[-1]): f.name for f in dataset.frames}
    requested = [args.start_id + i * args.stride for i in range(args.count)]
    missing = set(requested) - set(names)
    if missing:
        raise FileNotFoundError(f"Missing acquisition IDs: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(names[i] for i in requested) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
