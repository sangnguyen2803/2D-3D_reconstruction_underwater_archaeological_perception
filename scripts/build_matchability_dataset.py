"""Build all-view v3 matchability labels and E0 threshold/eligibility audit."""

import argparse
from pathlib import Path

from underwater_vision.matchability.build import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/matchability_labels_v3"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--sequential-only", action="store_true")
    args = p.parse_args()
    build_dataset(args.output, args.workers, args.window, not args.sequential_only)


if __name__ == "__main__":
    main()
