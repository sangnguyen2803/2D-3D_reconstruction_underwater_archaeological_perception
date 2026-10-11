"""Aggregate measured v3 seed/fold results and evaluate frozen Q1-Q5 criteria."""

import argparse
from pathlib import Path

from underwater_vision.matchability.report import aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.suite, a.output)


if __name__ == "__main__":
    main()
