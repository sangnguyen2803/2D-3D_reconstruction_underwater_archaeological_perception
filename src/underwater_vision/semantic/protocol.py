"""Versioned artifacts, human review gates and reproducible spatial manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_run(path):
    path = Path(path)
    if not path.name.endswith("_v2"):
        raise ValueError("Semantic experiment directories must end in _v2")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_manifest(path, require_frozen=False):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    ids = [e["image_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image ids in label manifest")
    for entry in entries:
        role = entry["role"]
        spatial = manifest.get("spatial_split", {}).get("groups")
        if spatial is not None and entry["image_id"] not in spatial.get(role, []):
            raise ValueError("Image id lies outside its recorded spatial group")
        if role in {"validation", "test"} and entry["slab"] != role:
            raise ValueError("Hand-label role does not match v1 spatial slab")
        if role == "train" and entry["slab"] != "train":
            raise ValueError("Training label outside train slab")
        if entry["label_source"] == "hand" and entry.get("reviewed"):
            if not entry.get("annotator") or not entry.get("date") or not entry.get("pass"):
                raise ValueError("Hand labels need annotator, UTC date and pass")
            mask = Path(entry["mask"])
            if not mask.is_file() or digest_file(mask) != entry.get("mask_sha256"):
                raise ValueError("Missing or modified reviewed hand label")
    if require_frozen:
        if not manifest.get("spatial_split"):
            raise ValueError("A frozen manifest requires the reference spatial partition")
        frozen = manifest.get("frozen")
        if not frozen or not all(
            e.get("reviewed") for e in entries if e["role"] in {"validation", "test"}
        ):
            raise ValueError("Hand validation/test labels must be reviewed and manifest frozen")
        canonical = json.dumps(entries, sort_keys=True).encode()
        if frozen["entries_sha256"] != hashlib.sha256(canonical).hexdigest():
            raise ValueError("Frozen manifest has changed")
        for entry in entries:
            if not Path(entry["mask"]).is_file() or digest_file(entry["mask"]) != entry.get(
                "mask_sha256"
            ):
                raise ValueError("Frozen mask content has changed")
    return manifest


def require_gate(path):
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("status") != "passed_human_review" or not gate.get("reviewer"):
        raise ValueError("Gate 0 needs human review of class separability and dynamic instances")
    return gate


def stratified(names, count):
    if count > len(names):
        raise ValueError("Not enough spatial views")
    return [names[i] for i in np.linspace(0, len(names) - 1, count, dtype=int)]


def spatial_folds(names, centers, gap=0.4471966):
    """Three geometric blocks, rotated test/val/train with boundary exclusions.

    These are independent of v1 slabs. Every learned module must be retrained for
    a fold; the v1 teacher/student cannot be reused as a no-leakage fold model.
    """
    centers = np.asarray(centers)
    origin = centers.mean(0)
    axis = np.linalg.svd(centers - origin, full_matrices=False)[2][0]
    position = (centers - origin) @ axis
    edges = np.quantile(position, [1 / 3, 2 / 3])
    block = np.digitize(position, edges)
    excluded = np.min(np.abs(position[:, None] - edges), axis=1) < gap / 2
    result = []
    for fold in range(3):
        groups = {"train": [], "validation": [], "test": [], "excluded": []}
        for i, name in enumerate(names):
            role = (
                "excluded"
                if excluded[i]
                else (
                    "test"
                    if block[i] == fold
                    else "validation" if block[i] == (fold + 1) % 3 else "train"
                )
            )
            groups[role].append(name)
        result.append(
            {
                "fold": fold,
                "groups": groups,
                "origin": origin.tolist(),
                "axis": axis.tolist(),
                "boundaries": edges.tolist(),
                "gap": gap,
                "specification": {
                    "method": "rotated_three_blocks_v2",
                    "fold": fold,
                    "origin": origin.tolist(),
                    "axis": axis.tolist(),
                    "boundaries": edges.tolist(),
                    "gap": gap,
                },
            }
        )
    return result
