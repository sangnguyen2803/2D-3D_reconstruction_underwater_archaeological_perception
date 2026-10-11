"""Finalize human Gate 0 or freeze hand labels after the repeat consistency pass."""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from underwater_vision.evaluation.segmentation import confusion, segmentation_metrics
from underwater_vision.semantic.protocol import digest_file, load_manifest, require_gate
from underwater_vision.utils.io import write_json


def finalize_gate(a):
    audit = json.loads((a.audit / "audit.json").read_text())
    with (a.audit / "human_review.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 150 or any(
        r["reviewed"] != "yes" or not r["reviewer"] or r["dynamic_instances"] == "" for r in rows
    ):
        raise ValueError(
            "All >=150 audit images need named human review and dynamic-instance counts"
        )
    if (a.audit / "gate_decision_v2.json").exists():
        raise FileExistsError("Gate decision already finalized")
    counts = [int(r["dynamic_instances"]) for r in rows]
    totals = np.sum(list(audit["pseudo_pixel_counts"].values()), axis=0)
    rare = np.flatnonzero(totals[:6] / totals.sum() < 0.01).tolist()
    merges = json.loads(a.merge_map)
    class_decisions = json.loads(a.class_decisions)
    from underwater_vision.semantic.schema import merge_mask

    merge_mask(np.arange(6, dtype=np.uint8)[None], merges)
    if any(str(c) not in merges and str(c) not in class_decisions for c in rare):
        raise ValueError(
            f"Rare teacher classes {rare} need explicit --merge-map or --class-decisions; distinguish actual rarity from teacher failure"
        )
    separable = all(r["classes_separable"] == "yes" for r in rows)
    fraction = np.mean(np.asarray(counts) > 0)
    detector = bool(fraction >= 0.05 and sum(counts) >= 100)
    decision = {
        "status": "passed_human_review" if separable else "failed_class_separability",
        "reviewer": sorted({r["reviewer"] for r in rows}),
        "date": datetime.now(UTC).isoformat(),
        "classes_separable": separable,
        "dynamic_image_fraction": float(fraction),
        "dynamic_instances": sum(counts),
        "detector": (
            "eligible_requires_box_annotation_and_implementation"
            if detector
            else "skip_insufficient_instances"
        ),
        "q2": "quantitative_eligible" if detector else "qualitative_only",
        "human_review_sha256": digest_file(a.audit / "human_review.csv"),
        "pseudo_statistics": audit["pseudo_pixel_counts"],
        "merge_map": json.loads(a.merge_map),
        "rare_class_decisions": class_decisions,
        "rare_classes": "Low-frequency classes require a documented reviewer merge/ignore decision; no automatic anatomical assumption",
    }
    write_json(a.audit / "gate_decision_v2.json", decision)
    write_json(a.audit / "gate_review.json", decision)
    if separable:
        criteria = yaml.safe_load(a.criteria.read_text())
        criteria.update(
            frozen=True,
            status="frozen_after_human_gate_review",
            gate_decision_sha256=digest_file(a.audit / "gate_decision_v2.json"),
        )
        a.criteria.write_text(yaml.safe_dump(criteria, sort_keys=False), encoding="utf-8")
        schema_path = Path("configs/semantic/labels.yaml")
        schema = yaml.safe_load(schema_path.read_text())
        schema.update(status="frozen_after_human_gate_review", merge_map=json.loads(a.merge_map))
        schema_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    print(decision)


def freeze_labels(a):
    require_gate(a.audit / "gate_review.json")
    manifest = load_manifest(a.manifest)
    if manifest.get("frozen"):
        raise FileExistsError("Manifest already frozen")
    for role, count in [("validation", 20), ("test", 40)]:
        entries = [e for e in manifest["entries"] if e["role"] == role]
        if len(entries) < count or any(not e.get("reviewed") for e in entries):
            raise ValueError(f"Need >= {count} reviewed hand labels for {role}")
    repeated = manifest.get("repeat_annotations", {})
    if len(manifest["repeat_tasks"]) < 8:
        raise ValueError("Need eight repeat annotations")
    rows = []
    for name in manifest["repeat_tasks"]:
        first = next(e for e in manifest["entries"] if e["image_id"] == name)
        second = repeated.get(name)
        if (
            not second
            or second["annotator"] != first["annotator"]
            or (
                datetime.fromisoformat(second["date"]) - datetime.fromisoformat(first["date"])
            ).total_seconds()
            < 48 * 3600
        ):
            raise ValueError("Same annotator must repeat each task after at least 48 hours")
        if digest_file(second["mask"]) != second["mask_sha256"]:
            raise ValueError("Repeat mask changed after review")
        matrix = confusion(
            np.asarray(Image.open(first["mask"])), np.asarray(Image.open(second["mask"]))
        )
        rows.append({"image_id": name, **segmentation_metrics(matrix)})
    for e in manifest["entries"]:
        if e["role"] == "train":
            if not Path(e["mask"]).is_file():
                raise ValueError("Pseudo train masks must exist before manifest freeze")
            e["mask_sha256"] = digest_file(e["mask"])
    manifest["frozen"] = {
        "date": datetime.now(UTC).isoformat(),
        "entries_sha256": hashlib.sha256(
            json.dumps(manifest["entries"], sort_keys=True).encode()
        ).hexdigest(),
        "repeat_consistency": rows,
    }
    write_json(a.manifest, manifest)
    load_manifest(a.manifest, require_frozen=True)
    print("Frozen label manifest with independent repeat consistency measurements")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["gate", "labels"])
    p.add_argument("--audit", type=Path, default=Path("outputs/semantic_audit_v2"))
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--criteria", type=Path, default=Path("configs/semantic/criteria_v2.yaml"))
    p.add_argument("--merge-map", default="{}", help='Reviewer decision JSON, e.g. {"3":255}')
    p.add_argument(
        "--class-decisions",
        default="{}",
        help="Human rationale for retaining/relabeling each rare class not merged; keyed by class id",
    )
    a = p.parse_args()
    finalize_gate(a) if a.mode == "gate" else freeze_labels(a)


if __name__ == "__main__":
    main()
