"""Resume checks must not reuse changed checkpoints, splits or incomplete predictions."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_segformer_folds import verified_run
from underwater_vision.matchability.fold_report import evaluate
from underwater_vision.matchability.protocol import digest
from underwater_vision.matchability.report import BASELINES
from underwater_vision.utils.io import write_json


def dense_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = Path("dataset")
    split = {
        "name": "fold0",
        "groups": {"train": ["a"], "validation": ["b"], "test": ["c"], "excluded": []},
    }
    write_json(dataset / "manifest_v3.json", {"protocols": [split]})
    criteria = Path("configs/matchability/criteria_v3.yaml")
    criteria.parent.mkdir(parents=True)
    criteria.write_text("status: frozen_after_E0\n")
    training = Path("configs/matchability/train.yaml")
    training.write_text("epochs: 4\n")
    folder = Path("run_v3")
    folder.mkdir()
    (folder / "model_v3.pt").write_bytes(b"original weights")
    write_json(
        folder / "evaluation_v3.json",
        {"status": "complete", "checkpoint_hash": digest(folder / "model_v3.pt")},
    )
    write_json(
        folder / "resolved_config_v3.json",
        {
            "protocol": "fold0",
            "seed": 42,
            "epochs": 4,
            "variant": "l1",
            "graph": "hybrid",
            "manifest_sha256": digest(dataset / "manifest_v3.json"),
            "criteria_sha256": digest(criteria),
            "model_config": {"kind": "segformer_b0"},
            "train_config": yaml.safe_load(training.read_text()),
            "spatial_split": split,
            "test_images": ["c"],
        },
    )
    (folder / "predictions").mkdir()
    (folder / "predictions/c.npz").write_bytes(b"prediction exists")
    return folder, dataset


def test_verified_completed_dense_run_can_be_reused(tmp_path, monkeypatch):
    folder, dataset = dense_run(tmp_path, monkeypatch)
    assert verified_run(folder, "segformer_b0", "fold0", 42, dataset)


def test_changed_checkpoint_cannot_be_reused(tmp_path, monkeypatch):
    folder, dataset = dense_run(tmp_path, monkeypatch)
    (folder / "model_v3.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_run(folder, "segformer_b0", "fold0", 42, dataset)


def test_incomplete_prediction_population_cannot_be_reused(tmp_path, monkeypatch):
    folder, dataset = dense_run(tmp_path, monkeypatch)
    (folder / "predictions/c.npz").unlink()
    with pytest.raises(ValueError, match="Missing or additional predictions"):
        verified_run(folder, "segformer_b0", "fold0", 42, dataset)


def test_another_seed_cannot_be_reused(tmp_path, monkeypatch):
    folder, dataset = dense_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="frozen protocol"):
        verified_run(folder, "segformer_b0", "fold0", 43, dataset)


def test_interrupted_run_is_not_considered_complete(tmp_path):
    assert not verified_run(tmp_path, "segformer_b0", "fold0", 42, tmp_path)


def test_complete_fold_report_reads_prediction_subfolders(tmp_path, monkeypatch):
    """Exercise collection, paired statistics and figures on all 18 tiny runs."""
    from underwater_vision.evaluation.matchability import metrics

    monkeypatch.chdir(tmp_path)
    dataset, suite = Path("dataset"), Path("suite_v3")
    names = ["image_a", "image_b"]
    labels = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8])
    dense_metrics = metrics(labels, probability)
    control_metrics = metrics(labels, 1 - probability)
    splits = [
        {"name": f"fold{fold}", "groups": {"train": [], "validation": [], "test": names}}
        for fold in range(3)
    ]
    write_json(dataset / "manifest_v3.json", {"protocols": splits})
    criteria = Path("configs/matchability/criteria_v3.yaml")
    criteria.parent.mkdir(parents=True)
    criteria.write_text("status: frozen\n")
    runs = []
    for fold in range(3):
        label_root = dataset / "labels" / f"fold{fold}" / "hybrid"
        label_root.mkdir(parents=True)
        for name in names:
            np.savez(label_root / f"{name}.npz", l1=labels)
        for seed in [42, 43, 44]:
            for kind in ["segformer_b0", "baselines"]:
                root = suite / f"{kind}_fold{fold}_s{seed}_v3"
                predictions = root / "predictions"
                predictions.mkdir(parents=True)
                metric = (
                    {
                        "test_after": dense_metrics,
                        "test_before": dense_metrics,
                        "temperature": 1,
                        "best_validation_ap": 1,
                        "history": [],
                    }
                    if kind == "segformer_b0"
                    else {"models": {name: control_metrics for name in BASELINES}}
                )
                write_json(root / "evaluation_v3.json", {"status": "complete", **metric})
                for name in names:
                    scores = (
                        {"probability": probability}
                        if kind == "segformer_b0"
                        else {key: 1 - probability for key in BASELINES}
                    )
                    np.savez(predictions / f"{name}.npz", name=name, labels=labels, **scores)
                runs.append(
                    {"kind": kind, "fold": fold, "seed": seed, "path": str(root), "reused": False}
                )
    write_json(
        suite / "fold_manifest_v3.json",
        {
            "dataset": str(dataset),
            "dataset_manifest_sha256": digest(dataset / "manifest_v3.json"),
            "criteria_sha256": digest(criteria),
            "runs": runs,
            "scope": "test",
        },
    )
    # Digests in the report refer to real source files; keep them resolvable after chdir.
    from underwater_vision.matchability import fold_report

    real_digest = digest
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        fold_report,
        "digest",
        lambda p: real_digest(repo / p if Path(p).parts[0] == "src" else p),
    )
    monkeypatch.setattr(
        fold_report,
        "load_features",
        lambda *_: {
            "statistics": np.ones((4, 3)),
            "xy": np.ones((4, 2)),
            "sharpness": np.ones(4),
            "brightness": 1,
        },
    )
    report = evaluate(suite, "comparison_v3", "report.md", "figures_v3")
    assert report["criteria"]["Q1"]["positive_CI_folds"] == ["fold0", "fold1", "fold2"]
    assert len(report["run_provenance"]) == 18
    assert len(list(Path("figures_v3").glob("*.png"))) == 4
    assert Path("report.md").is_file()
