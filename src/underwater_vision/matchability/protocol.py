"""Immutable v3 runs, provenance and leakage checks; no semantic-label gates."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_run(path):
    path = Path(path)
    if not path.name.endswith("_v3"):
        raise ValueError("New matchability result directories must end in _v3")
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_evaluation(metadata, names, split, role="test"):
    names = set(names)
    fitted = set(metadata["training_images"] + metadata["validation_images"])
    if names & fitted:
        raise ValueError("Matchability training/validation image leakage")
    if not names.issubset(set(split["groups"][role])):
        raise ValueError("Matchability evaluation images outside the locked spatial slab")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
