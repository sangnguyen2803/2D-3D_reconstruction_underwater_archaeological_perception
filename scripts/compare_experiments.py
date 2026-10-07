"""Build measured comparison tables; reject comparisons with different inputs."""

import argparse
import json
from pathlib import Path

from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/comparison"))
    args = p.parse_args()
    rows, selection = [], None
    for run in args.runs:
        manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
        names = [entry["name"] for entry in manifest]
        if selection is not None and names != selection:
            raise ValueError("Controlled comparison requires identical image names and ordering")
        selection = names
        if (run / "metrics.json").exists():
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        else:
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            if "registered no model" not in status.get("error", ""):
                raise ValueError(f"Run incomplete for an unsupported reason: {run}")
            pairs = json.loads((run / "pair_metrics.json").read_text(encoding="utf-8"))
            total = sum(p["matches"] for p in pairs)
            metrics = {
                "registered_images": 0,
                "sparse_points": 0,
                "sparse_status": "failed",
                "matches": total,
                "inliers": sum(p["inliers"] for p in pairs),
                "inlier_ratio": sum(p["inliers"] for p in pairs) / max(total, 1),
                "failure_reason": status["error"],
            }
        rows.append({"run": run.name, **metrics})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", rows)
    columns = [
        "run",
        "registered_images",
        "sparse_points",
        "inlier_ratio",
        "mean_reprojection_error_px",
        "mean_track_length",
    ]
    table = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in rows:
        table += (
            "| "
            + " | ".join(
                (
                    f'{row.get(c, "N/A"):.4f}'
                    if isinstance(row.get(c), float)
                    else str(row.get(c, "N/A"))
                )
                for c in columns
            )
            + " |\n"
        )
    (args.output / "comparison.md").write_text(
        "# Measured experiment comparison\n\n" + table, encoding="utf-8"
    )
    print(table)


if __name__ == "__main__":
    main()
