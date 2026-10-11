"""Generate measured SegFormer/baseline three-fold comparisons and a new report."""

import argparse
from pathlib import Path

from underwater_vision.matchability.fold_report import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=Path("docs/segformer_folds_report_v3.md"))
    p.add_argument("--figures", type=Path, default=Path("docs/figures_v3/segformer_folds_v3"))
    a = p.parse_args()
    evaluate(a.suite, a.output, a.report, a.figures)


if __name__ == "__main__":
    main()
