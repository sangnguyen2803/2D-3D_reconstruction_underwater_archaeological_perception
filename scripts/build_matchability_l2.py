"""Build the stronger reference-triangulation ablation without overwriting L1."""

import argparse
from pathlib import Path

from underwater_vision.matchability.strong_labels import build_strong_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build_strong_labels(a.dataset, a.output)


if __name__ == "__main__":
    main()
