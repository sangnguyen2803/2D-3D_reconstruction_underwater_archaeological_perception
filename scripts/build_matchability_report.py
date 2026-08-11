"""Create new English v3 cards, summary, report and measured figures."""

import argparse
from pathlib import Path

from underwater_vision.matchability.documentation import build_documents


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--comparison", type=Path, default=Path("outputs/comparison_v3/comparison_v3.json")
    )
    p.add_argument(
        "--verification", type=Path, default=Path("outputs/verification_v3/verification_v3.json")
    )
    p.add_argument("--skip-heatmaps", action="store_true")
    a = p.parse_args()
    build_documents(a.comparison, a.verification, not a.skip_heatmaps)


if __name__ == "__main__":
    main()
