"""Export the standalone student and measure parity/latency on local hardware."""

import argparse
from pathlib import Path

from underwater_vision.serving.export import export_onnx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/deployment_v2"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--task", choices=["semantic", "matchability"], default="semantic")
    p.add_argument(
        "--skip-timing", action="store_true", help="Export/parity only while GPU training is active"
    )
    a = p.parse_args()
    if a.task == "matchability":
        from underwater_vision.serving.matchability_export import export_matchability

        export_matchability(a.checkpoint, a.output, a.batch, measure=not a.skip_timing)
    else:
        export_onnx(a.checkpoint, a.output, a.batch)


if __name__ == "__main__":
    main()
