"""Gate 0 proposals on >=150 spatially stratified views; human review is required."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.semantic.protocol import new_run, spatial_folds, stratified
from underwater_vision.semantic.pseudo_labels import GroundedSAMTeacher
from underwater_vision.semantic.schema import DYNAMIC, NAMES, colorize
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json, write_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_audit_v2"))
    p.add_argument("--sample", type=int, default=150)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished proposals; never replace completed audit",
    )
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.sample < 150:
        raise ValueError("Gate 0 requires at least 150 views")
    if args.resume:
        root = args.output
        if not root.exists() or (root / "audit.json").exists():
            raise ValueError("Resume requires an unfinished audit")
    else:
        root = new_run(args.output)
    split = json.loads(Path("outputs/reference_audit_v1/spatial_split_v1.json").read_text())
    lookup = {
        f.name: f for f in MermaidDataset("data/raw/mermaid").frames if f.reference_c2w is not None
    }
    groups = {
        s: stratified(split["groups"][s], min(len(split["groups"][s]), args.sample // 3))
        for s in ["train", "validation", "test"]
    }
    if sum(map(len, groups.values())) < 150:
        raise ValueError("Insufficient spatial audit coverage")
    for folder in ["images", "teacher", "overlays"]:
        (root / folder).mkdir(exist_ok=True)
    write_json(
        root / "sample_manifest.json",
        {"groups": groups, "split": split, "sample": sum(map(len, groups.values()))},
    )
    frames = list(lookup.values())
    write_json(
        root / "spatial_folds_v2.json",
        {
            "folds": spatial_folds(
                [f.name for f in frames],
                [f.reference_c2w[:3, 3] for f in frames],
                split["specification"]["gap"],
            ),
            "requires_fold_specific_training": True,
        },
    )
    teacher = None
    entries = []
    histograms = {s: np.zeros(7, np.int64) for s in groups}
    for slab, names in groups.items():
        for name in names:
            stem = Path(name).stem
            record = root / "teacher" / (stem + ".npz")
            proposal_path = root / "teacher" / (stem + ".json")
            if record.exists() and proposal_path.exists():
                with np.load(record) as z:
                    mask = z["mask"]
                proposals = json.loads(proposal_path.read_text())["proposals"]
            else:
                if teacher is None:
                    teacher = GroundedSAMTeacher(args.device)
                    write_json(root / "teacher" / "teacher_provenance.json", teacher.provenance)
                rgb, _ = resize_rgb(read_rgb(lookup[name].path), 960)
                result = teacher.predict(rgb)
                mask, proposals = result["mask"], result["proposals"]
                atomic_npz(record, mask=mask, confidence=result["confidence"])
                write_json(proposal_path, {"image_id": name, "proposals": proposals})
                write_rgb(root / "images" / name, rgb)
                write_rgb(
                    root / "overlays" / name,
                    np.clip(0.6 * rgb + 0.4 * colorize(mask), 0, 255).astype(np.uint8),
                )
                Image.fromarray(mask).save(root / "teacher" / (stem + ".png"))
            bins = np.bincount(np.where(mask == 255, 6, mask).ravel(), minlength=7)
            histograms[slab] += bins
            entries.append(
                {
                    "image_id": name,
                    "slab": slab,
                    "proposed_dynamic_instances": sum(
                        p["class"] == DYNAMIC and p.get("accepted", False) for p in proposals
                    ),
                    "proposed_dynamic_pixels": int((mask == DYNAMIC).sum()),
                }
            )
            print(
                f"Gate 0 proposals {len(entries)}/{sum(map(len,groups.values()))}: {name}",
                flush=True,
            )
    # Match endpoints are measured only on audited images; missing coverage is explicit.
    endpoint_total = dynamic_endpoints = 0
    audited = {e["image_id"] for e in entries}
    cache_features = {}
    from underwater_vision.features.dinov2 import DinoExtractor
    from underwater_vision.features.store import feature_record

    neural = None
    for file in Path("outputs/match_filter_v1/pairs").glob("*.npz"):
        with np.load(file) as z:
            a, b, pairs = str(z["a"]), str(z["b"]), z["pairs"]
        for name, column in [(a, 0), (b, 1)]:
            if name not in audited:
                continue
            if name not in cache_features:
                if neural is None:
                    # Release foundation models before loading the v1 feature model.
                    teacher = None
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    neural = DinoExtractor(
                        "dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
                    )
                cache_features[name] = feature_record(lookup[name], neural)
            xy = cache_features[name]["xy"][pairs[:, column]]
            with np.load(root / "teacher" / (Path(name).stem + ".npz")) as z:
                mask = z["mask"]
            xy = np.round(xy).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
            endpoint_total += len(xy)
            dynamic_endpoints += int((mask[xy[:, 1], xy[:, 0]] == DYNAMIC).sum())
    with (root / "human_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "slab",
                "proposed_dynamic_instances",
                "proposed_dynamic_pixels",
                "reviewed",
                "dynamic_instances",
                "classes_separable",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)
    audit = {
        "status": "awaiting_human_review",
        "sample": len(entries),
        "slab_counts": {s: len(n) for s, n in groups.items()},
        "class_names": [*NAMES, "ignore"],
        "pseudo_pixel_counts": {s: h.tolist() for s, h in histograms.items()},
        "proposed_dynamic_images": sum(e["proposed_dynamic_instances"] > 0 for e in entries),
        "proposed_dynamic_instances": sum(e["proposed_dynamic_instances"] for e in entries),
        "audited_v1_endpoints": endpoint_total,
        "dynamic_endpoint_fraction": dynamic_endpoints / max(endpoint_total, 1),
        "endpoint_scope": "top1 v1 pair candidates whose image appears in audit; proposals only",
        "decisions": "pending human review; detector and class merges not decided",
        "hand_review": "human_review.csv",
    }
    write_json(root / "audit.json", audit)
    write_json(
        root / "gate_review.json",
        {"status": "pending", "reviewer": None, "classes_separable": None, "detector": "undecided"},
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (slab, h) in enumerate(histograms.items()):
        ax.bar(np.arange(7) + i * 0.25, h / max(h.sum(), 1), width=0.25, label=slab)
    ax.set_xticks(np.arange(7) + 0.25, [*NAMES, "ignore"], rotation=20)
    ax.set_ylabel("Pseudo-label pixel fraction (unreviewed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "audit_histograms_v2.png", dpi=160)


if __name__ == "__main__":
    main()
