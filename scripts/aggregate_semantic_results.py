"""Aggregate measured seed/fold results; pending criteria never count as success."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from underwater_vision.semantic.protocol import new_run
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_comparison_final_v2"))
    a = p.parse_args()
    plan = json.loads((a.suite / "plan_v2.json").read_text())
    thresholds = plan.get("criteria") or yaml.safe_load(
        Path("configs/semantic/criteria_v2.yaml").read_text()
    )
    root = new_run(a.output)
    rows = {}
    for path in a.suite.glob("*/evaluation_v2.json"):
        rows[path.parent.name] = json.loads(path.read_text())
    for path in a.suite.glob("*/metrics.json"):
        rows[path.parent.name] = json.loads(path.read_text())
    # Summaries retain separate fold/seed entries so fewer runs cannot masquerade
    # as the pre-registered 3x3 protocol.
    groups = {}
    for name, data in rows.items():
        prefix = name.split("_", 2)
        if len(prefix) < 3:
            continue
        group = prefix[2]
        groups.setdefault(group, []).append({"fold": prefix[0], "seed": prefix[1], "data": data})
    summary = {}
    for name, items in groups.items():
        values = {}
        for field in [
            "miou",
            "pixel_accuracy",
            "sparse_points",
            "registered_images",
            "mean_reprojection_error_px",
        ]:
            observed = [row["data"][field] for row in items if row["data"].get(field) is not None]
            if observed:
                values[field] = {
                    "mean": float(np.mean(observed)),
                    "std": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
                    "runs": len(observed),
                }
        summary[name] = {
            "runs": len(items),
            "spatial_regions": len({row["fold"] for row in items}),
            "seeds": len({row["seed"] for row in items}),
            "metrics": values,
        }
    final = plan["final_protocol"]
    criteria = {
        q: {
            "status": "pending_or_incomplete",
            "reason": "Requires measured hand-label results and the complete pre-registered protocol",
        }
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]
    }
    student_rows = groups.get("segformer_b0_evaluation_v2", [])
    teacher_rows = groups.get("teacher_evaluation_v2", [])
    probe_rows = groups.get("dino_probe_evaluation_v2", [])
    if final and all(len(rows) == 9 for rows in [student_rows, teacher_rows, probe_rows]):
        mean_student = np.mean([r["data"]["miou"] for r in student_rows])
        mean_teacher = np.mean([r["data"]["miou"] for r in teacher_rows])
        mean_probe = np.mean([r["data"]["miou"] for r in probe_rows])
        gap = thresholds["Q1"]["teacher_miou_gap_max"]
        success = mean_student >= mean_teacher - gap and mean_student > mean_probe
        criteria["Q1"] = {
            "status": "met" if success else "not_met",
            "student_miou": float(mean_student),
            "teacher_miou": float(mean_teacher),
            "probe_miou": float(mean_probe),
            "gap_max": gap,
        }
    dynamic = groups.get("dynamic_evaluation_v2", [])
    if final and len(dynamic) == 9:
        success = all(
            r["data"].get("status") == "eligible"
            and r["data"].get("precision_differences", {}).get("student", {}).get("ci95")
            and r["data"]["precision_differences"]["student"]["ci95"][0]
            > thresholds["Q2"]["precision_delta_ci_lower_gt"]
            for r in dynamic
        )
        criteria["Q2"] = {
            "status": "met" if success else "not_met_or_qualitative",
            "rule": "all planned seed/fold student precision CIs exclude zero on verified dynamic pairs",
        }
    for mode in ["compact", "full"]:
        items = groups.get(f"match_eval_{mode}_v2", [])
        fold_success = []
        for fold in range(3):
            fold_rows = [r for r in items if r["fold"] == f"fold{fold}"]
            # Conservative criterion: all three seed CIs must exclude zero.
            fold_success.append(
                len(fold_rows) == 3
                and all(
                    r["data"]["ap_difference"]["ci95"]
                    and r["data"]["ap_difference"]["ci95"][0]
                    > thresholds["Q3"]["ap_delta_ci_lower_gt"]
                    for r in fold_rows
                )
            )
        if final and len(items) == 9:
            criteria["Q3_" + mode] = {
                "status": (
                    "met"
                    if sum(fold_success) >= thresholds["Q3"]["required_successful_folds"]
                    and criteria["Q1"]["status"] == "met"
                    else "not_met_or_Q1_gate_failed"
                ),
                "successful_folds": sum(fold_success),
                "rule": "AP CI lower>0 in all 3 seeds of at least 2 folds",
            }
    confidence = groups.get("confidence_v2", [])
    if final and len(confidence) == 9:
        success = all(
            r["data"]["ause_differences"][b]["ci95"]
            and r["data"]["ause_differences"][b]["ci95"][1]
            < thresholds["Q4"]["ause_delta_ci_upper_lt"]
            for r in confidence
            for b in ["heuristic", "v1_learned"]
        )
        criteria["Q4"] = {
            "status": (
                "met"
                if success and criteria["Q1"]["status"] == "met"
                else "not_met_or_Q1_gate_failed"
            ),
            "rule": "AUSE difference CI upper<0 against both baselines in all planned seed/fold runs",
        }
    voting = groups.get("voting_evaluation_v2", [])
    if final and len(voting) == 9:
        criteria["Q5"] = {
            "status": (
                "descriptive_reported"
                if all(r["data"].get("accuracy_bootstrap") for r in voting)
                else "insufficient_hand_observations"
            ),
            "threshold": None,
        }
    write_json(
        root / "comparison_v2.json",
        {
            "raw": rows,
            "summary": summary,
            "criteria": criteria,
            "planned_protocol": plan,
            "negative_runs_retained": True,
        },
    )


if __name__ == "__main__":
    main()
