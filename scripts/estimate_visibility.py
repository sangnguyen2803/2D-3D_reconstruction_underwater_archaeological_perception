import argparse
from pathlib import Path

import numpy as np

from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.utils.io import read_rgb
from underwater_vision.visibility.deterministic import estimate_visibility
from underwater_vision.visualization.figures import visibility_figure


def main():
    p = argparse.ArgumentParser(
        description="Estimate and visualize deterministic underwater reliability"
    )
    p.add_argument("image")
    p.add_argument("--output", default="outputs/visibility", type=Path)
    p.add_argument("--max-size", default=960, type=int)
    args = p.parse_args()
    image, _ = resize_rgb(read_rgb(args.image), args.max_size)
    visibility, cues = estimate_visibility(image)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "visibility.npz", reliability=visibility, **cues)
    visibility_figure(image, visibility, args.output / "visibility.png")


if __name__ == "__main__":
    main()
