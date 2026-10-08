"""Generate v1 reports and comparison plots without replacing original reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from underwater_vision.utils.io import write_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(columns, rows):
    return (
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join(["---"] * len(columns))
        + " |\n"
        + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
    )


def main():
    summary_path = Path("PROJECT_SUMMARY_v1.md")
    report_path = Path("docs/research_report_v1.md")
    status_path = Path("PROJECT_STATUS_v1.md")
    output = Path("outputs/comparison_v1")
    if any(p.exists() for p in [summary_path, report_path, status_path, output]):
        raise FileExistsError(
            "Existing v1 reports/results are immutable; select a new report version"
        )
    output.mkdir(parents=True)
    figures = Path("docs/figures_v1")
    figures.mkdir(parents=True, exist_ok=True)
    audit = load("outputs/reference_audit_v1/audit_v1.json")
    split = load("outputs/reference_audit_v1/spatial_split_v1.json")
    training = load("outputs/match_filter_v1/training_v1.json")
    matching = load("outputs/match_evaluation_v1/evaluation_v1.json")
    retrieval = load("outputs/retrieval_v1/metrics_v1.json")
    confidence = load("outputs/point_confidence_v1/evaluation_v1.json")
    runs = [
        "classical_eval",
        "neural_gpu_eval",
        "visibility_gpu_eval",
        "match_filter_sfm_v1",
        "mlp_filter_sfm_v1",
        "retrieval_filter_sfm_v1",
    ]
    sfm = {name: load(f"outputs/{name}/metrics.json") for name in runs}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / (name + "_v1.png"), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(audit["alternatives"]), [v["median_px"] for v in audit["alternatives"].values()])
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=18)
    ax.set_ylabel("Median reference epipolar error (pixels)")
    ax.set_title("Calibration and pose-convention sanity check")
    save(fig, "reference_audit")
    fig, ax = plt.subplots(figsize=(7, 5))
    for group in ["train", "validation", "test", "excluded"]:
        names = set(split["groups"][group])
        from underwater_vision.data.dataset_loader import MermaidDataset

        xyz = np.array(
            [
                f.reference_c2w[:3, 3]
                for f in MermaidDataset("data/raw/mermaid").frames
                if f.name in names
            ]
        )
        ax.scatter(xyz[:, 0], xyz[:, 1], s=5, label=group)
    ax.set(
        xlabel="Reference X",
        ylabel="Reference Y",
        title="Camera-center spatial blocks with excluded bands",
    )
    ax.legend()
    save(fig, "spatial_split")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, m in matching["models"].items():
        axes[0].plot(
            m["pr_curve"]["recall"],
            m["pr_curve"]["precision"],
            label=f"{name} AP={m['average_precision']:.3f}",
        )
        bins = m["calibration"]
        axes[1].plot(
            [b["predicted"] for b in bins], [b["observed"] for b in bins], marker=".", label=name
        )
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Reference-F labels: common proposal pool",
        ylim=(0, 1.02),
    )
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(
        xlabel="Score / predicted probability",
        ylabel="Reference-positive fraction",
        title="Match calibration",
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    save(fig, "match_evaluation")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Sequential ±6", "DINO top-12", "DINO long-range"],
        [retrieval["sequential_recall"], retrieval["recall"], retrieval["long_range_recall"]],
    )
    ax.set(
        ylabel="Recall of reference co-frustum proxy",
        title="Full 1,244-view retrieval; 12 candidates/query",
    )
    save(fig, "retrieval")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, m in confidence["models"].items():
        curve = m["curve"]
        axes[0].plot(
            curve["fraction_removed"],
            curve["retained_error_normalized"],
            label=f"{name} AUSE={m['ause']:.3f}",
        )
        bins = m["calibration"]
        axes[1].plot(
            [b["predicted_error_mean"] for b in bins],
            [b["observed_error_mean"] for b in bins],
            marker=".",
            label=name,
        )
    axes[0].plot(curve["fraction_removed"], curve["oracle_error_normalized"], "k--", label="Oracle")
    axes[0].set(
        xlabel="Fraction of uncertain points removed",
        ylabel="Retained mean error / initial mean",
        title="Held-out-view sparsification (1,678 points)",
    )
    axes[1].plot([0.01, 100], [0.01, 100], "k--")
    axes[1].set(
        xlabel="Predicted mean error, px",
        ylabel="Observed mean error, px",
        title="Error calibration (log scale)",
        xscale="log",
        yscale="log",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save(fig, "point_confidence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(runs, [sfm[n]["sparse_points"] for n in runs])
    axes[1].bar(runs, [sfm[n]["mean_reprojection_error_px"] for n in runs])
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
    axes[0].set(ylabel="Sparse points", title="Before/after: identical 16 images")
    axes[1].set(ylabel="Mean reprojection error, px", title="Internal fit: lower is better")
    save(fig, "sfm_comparison")
    distributions = {}
    for group in ["train", "validation", "test"]:
        with np.load(f"outputs/point_confidence_v1/{group}_v1.npz") as z:
            distributions[group] = dict(
                zip(
                    ["median", "p90", "p99", "maximum"],
                    np.quantile(z["error"], [0.5, 0.9, 0.99, 1]).tolist(),
                    strict=True,
                )
            )
    combined = {
        "audit": audit,
        "matching": matching,
        "retrieval": retrieval,
        "point_confidence": confidence,
        "sfm": sfm,
        "point_error_distributions": distributions,
        "validation": {"pytest": 35, "source": "local suite, not a remote CI claim"},
    }
    write_json(output / "comparison_v1.json", combined)
    audit_table = table(
        ["Convention", "Median px", "Fraction <=1.5px"],
        [
            [n, f"{v['median_px']:.4f}", f"{100*v['fraction_below_1_5_px']:.2f}%"]
            for n, v in audit["alternatives"].items()
        ],
    )
    match_table = table(
        ["Method", "AP", "ROC AUC", "Precision", "Recall", "F1"],
        [
            [n]
            + [f"{m[k]:.4f}" for k in ["average_precision", "roc_auc", "precision", "recall", "f1"]]
            for n, m in matching["models"].items()
        ],
    )
    sfm_table = table(
        ["Run", "Cameras", "Points", "Reprojection px", "ATE reference units"],
        [
            [
                n,
                sfm[n]["registered_images"],
                sfm[n]["sparse_points"],
                f"{sfm[n]['mean_reprojection_error_px']:.4f}",
                f"{sfm[n]['trajectory']['ate_rmse_reference_units']:.5f}",
            ]
            for n in runs
        ],
    )
    confidence_table = table(
        ["Method", "AUSE ↓", "Spearman(error) ↑", "Error MAE px"],
        [
            [n, f"{m['ause']:.4f}", f"{m['spearman_error']:.4f}", f"{m['prediction_mae_px']:.4f}"]
            for n, m in confidence["models"].items()
        ],
    )
    body = f"""# Project summary v1 — reference-supervised learning and independent component checks

This version adds three learned components and compares them with the original system. Original reports, summary and result directories remain unchanged. Code and configuration files are updated directly. All new experiment directories and this report carry `v1`.

The original detailed overview remains in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md). The current component measurements are saved in `outputs/comparison_v1/comparison_v1.json`.

## Executive assessment

Reference geometry passes a distributed calibration/pose sanity check. Learned match scoring improves AP/ROC AUC over ratio scoring on the spatial development holdout and restores sparse support compared with the former fused-descriptor visibility pipeline. It still raises reprojection error relative to the original baselines. Global DINO retrieval finds long-range candidates, but its overall overlap-proxy recall is below the sequential baseline at equal query budget. Learned point-error prediction is operational and independently measured; its present AUSE and interval coverage do not establish an improvement over heuristic confidence.

## Step 0: reference-camera geometry gate

All 1,244 published poses enter the spatial partition and retrieval geometry. The audit samples 128 pairs distributed over the full acquisition and checks 11,938 mutual, tight-ratio SIFT matches without estimating a RANSAC model. Reference F is computed for every subsequently labeled pair and all 10,306 retrieved unique pairs.

For camera-to-world poses, relative geometry is `inverse(T_b) @ T_a`, `E = skew(t) @ R`, and `F = inverse(K_b).T @ E @ inverse(K_a)`. Observed coordinates are Brown-undistorted before reference epipolar error is evaluated. The implementation supports both OpenCV 4 and OpenCV 5 iterative-undistortion interfaces.

{audit_table}

The gate requires at least 1,000 audited matches, median error <=1.5px and >=60% below 1.5px. Nominal median error is 0.2934px and the passing fraction is 94.14%. The large errors under inverse-pose and absolute-principal-point alternatives support the selected conventions.

Epipolar compatibility is a necessary geometric condition, not proof that a correspondence is correct. Published poses were fitted by BA on the dataset; they provide supervision independent of this project's RANSAC, not independently surveyed ground truth. The sanity check establishes consistency with the recorded assumptions, not a complete physical validation of underwater calibration.

![Calibration audit](docs/figures_v1/reference_audit_v1.png)

## Spatial partitions and leakage controls

The partition uses reference-camera position projected onto the principal scene axis. The original evaluation images anchor a test interval. Excluded bands separate test, training and validation regions. Counts are 226 train, 65 validation, 743 test and 210 excluded views; the spatial gap is 0.4472 reference units. Training selects 64 train and 32 validation views; the original 16-image list is used for the development comparison.

Camera-center separation does not guarantee disjoint visible surfaces. This limitation is retained explicitly. The 3D-confidence dataset additionally partitions triangulated reference XYZ: points outside their assigned spatial block are excluded. Normalization uses training data; model selection, probability temperature and decision threshold use validation. No model is fitted to test labels.

The 16-image region was already examined in the original project. These are spatial development holdout results, not a newly collected blind final benchmark. Multiple held-out scene regions and repeated seeds remain necessary.

![Spatial partitions](docs/figures_v1/spatial_split_v1.png)

## A: learned correspondence filtering

Input per top-1 RootSIFT candidate comprises descriptor distance, nearest-neighbor ratio, mutuality, DINO cosine, endpoint image quality, normalized endpoint coordinates and displacement (12 values). Candidates are generated before ratio filtering so the learner sees negative proposals.

Reference-F error <=1.5px is a positive compatibility label; error >=4px is negative. The ambiguous band is excluded from the loss and metrics. Training has {training['pair_counts']['train']} pairs; validation has {training['pair_counts']['validation']} pairs. Evaluation has 75 pairs and 82,500 known common-pool labels.

Both an independent-match MLP and a custom contextual attention network are implemented. The context model pools the complete pair into eight latent tokens and attends back to each match, preserving permutation equivariance while avoiding a full quadratic match-attention matrix. This is a compact custom model inspired by contextual correspondence filtering, not a reproduction of OANet.

Training uses BCE, AdamW, 20 epochs, gradient clipping and seed 42. Temperature and the F1 operating threshold are fitted only on spatial validation. The context architecture is selected by validation AP; its threshold is approximately 0.8274. The MLP remains an ablation even though its test AP is slightly higher.

{match_table}

AP is average precision, and AUC is ROC AUC. Ratio and old heuristic scores are ranking baselines rather than calibrated probabilities. The old-visibility row evaluates its fixed fusion/quality scoring before custom weighted RANSAC. The full old reconstruction is compared below using preserved artifacts. LightGlue uses official pretrained SIFT weights and the same keypoints, RootSIFT descriptors, scales and orientations. Its source revision is pinned to `eb42fee2d71449efb0aa5c10549752b5d75384d8`.

All aligned metrics share the RootSIFT top-1 proposal pool. LightGlue or fused matches outside this pool are excluded from aligned PR/AUC; LightGlue's native accepted-match precision is recorded separately. Different proposal coverage must be considered when reading these rows. LightGlue achieves the strongest common-pool F1 here; the custom networks have stronger score-ranking AP/AUC.

The new SfM path applies learned probability thresholds and resolves competing feature assignments. It does not use fixed descriptor fusion alpha or the custom weighted RANSAC. COLMAP then performs its standard verification and BA. Probability changes which matches enter COLMAP; its BA objective remains unweighted.

![Match evaluation](docs/figures_v1/match_evaluation_v1.png)

## B: DINO global retrieval

Frozen B/14 CLS descriptors are extracted at a longest edge of 336px for all 1,244 images. Cosine retrieval selects 12 neighbors per query, yielding 10,306 unique pairs. Inference uses images only. Reference poses are used afterward for evaluation, never as retrieval features.

Reference poses alone cannot determine true overlap without scene depth, relief and occlusion. The current evaluator therefore reports a **co-frustum proxy**: a 5x5 view grid at median reference-triangulated depth (2.8184 reference units), with reciprocal in-frustum coverage >=25%. Six sensitivity settings vary depth by factors 0.5/1/2 and overlap threshold 25%/50%.

At the equal nominal budget of 12 candidates/query, sequential ±6 recall is **10.08%** and DINO top-12 recall is **9.04%** against the default proxy. DINO long-range recall is **4.58%** for positive pairs separated by more than six indices; that candidate class is absent from the sequential window. Default DINO proxy precision is **87.06%**. There are many proxy positives per query, so recall is budget-limited and strongly dependent on assumed depth.

The recommended experimental option is hybrid retrieval plus sequential edges. On the original 16 images it supplies 114 pairs instead of 75; this has a larger pair budget, so it is not an equal-cost comparison. A full-scene SfM benchmark has not been run in this revision. Full-scene global retrieval and its candidate/reference-F graph have been completed.

![Retrieval comparison](docs/figures_v1/retrieval_v1.png)

## C: learned point error and confidence

The point learner predicts a Gaussian distribution for `log1p(held-out reprojection error)`. Inputs describe observed-view count, triangulation angles, training-view fit error, endpoint quality and DINO consistency/variation across the observed track. Track length and unique view count coincide for these conflict-free tracks, so they are represented by a single count feature.

For each track, a deterministic view is held out. The point is retriangulated with published cameras using the remaining views only. The held-out measurement supplies the target error. Its coordinates, descriptor and quality never enter learner inputs; a regression test verifies this exclusion. Training/validation tracks are constructed with ratio matching and RANSAC, but the target is measured with reference cameras and the excluded view, rather than copying a RANSAC mask or fitted-view residual.

After geometry/spatial exclusions, the dataset has **260 train, 127 validation and 1,678 test points**. Test tracks come from the original classical reconstruction, holding the point population fixed for comparing confidence methods. Geometry-invalid and spatially excluded counts are recorded, not silently counted as correct.

The MLP is 8->48->32->2 with GELU. Training uses 150 epochs, AdamW and log-error Gaussian NLL. A scale is fitted on validation only. Heuristic and view-count baselines receive validation-only error-unit scaling for calibration/MAE; their ranking is unchanged by that scaling.

{confidence_table}

Lower AUSE and higher Spearman(error) are preferable. Sparsification removes the points with highest predicted uncertainty first. Curves show retained mean error normalized by the initial mean. AUSE integrates its difference from an oracle over removal fractions 0 to 0.99. Tied uncertainties are averaged within groups, so the view-count baseline cannot gain from favorable input ordering.

The learned model currently has **AUSE 0.4910**, worse than the heuristic's **0.4194**. Its nominal 90% interval covers only **36.59%** of test errors. This is evidence against claiming calibrated uncertainty or superior ranking in v1. Mean held-out test error is 1.1386px and the median is 0.8715px.

Validation contains extreme error outliers (maximum about 3,083px), while the test maximum is about 9.74px. This distribution shift and limited training support strongly affect NLL/scale fitting. They also explain why least-squares-scaled simple baselines have poor error-unit MAE despite more useful rankings. Robust calibration, better-supported training tracks and additional spatial blocks are priorities.

Production confidence export uses estimated SfM cameras, observed descriptors and quality only; reference poses are not used for prediction. The combined v1 reconstruction has learned confidence/error/interval fields for all 1,912 sparse points. These outputs remain experimental evidence scores.

![Point uncertainty evaluation](docs/figures_v1/point_confidence_v1.png)

## Before/after SfM on identical input

All rows use the original 16 images, 960x720 resolution, seed 42 and the same calibration/mapper settings. The first three rows are preserved pre-change runs. New rows use learned probability filtering. Only the combined row changes pair selection to hybrid.

{sfm_table}

Context filtering on the unchanged 75-pair window increases sparse support from 1,065 to 1,890 points versus old visibility (about +77.46%), and from 1,790 to 1,890 versus classical (+5.59%). Mean reprojection error rises to 0.4504px. Hybrid retrieval adds 22 more points (1,912 total) and slightly lowers that internal error to 0.4474px. All variants retain 16 cameras. Neither result establishes dominant geometric accuracy.

No new dense MVS claim is made: these v1 runs stop at sparse reconstruction and learned point confidence. The existing dense clouds/meshes are retained. The v1 path still supports optional dense processing.

![SfM comparison](docs/figures_v1/sfm_comparison_v1.png)

## Reproduction and artifacts

Install the research extra with `python -m pip install -e ".[neural,research,dev]"`. The optional LightGlue baseline also needs Kornia and the pinned official LightGlue source. In the prepared environment both are installed and pretrained SIFT LightGlue weights are cached.

```powershell
.\\.venv\\Scripts\\python.exe scripts/audit_reference.py
.\\.venv\\Scripts\\python.exe scripts/retrieve_pairs.py
.\\.venv\\Scripts\\python.exe scripts/train_match_filter.py
.\\.venv\\Scripts\\python.exe scripts/evaluate_match_filter.py
.\\.venv\\Scripts\\python.exe scripts/train_point_confidence.py
.\\.venv\\Scripts\\python.exe scripts/run_experiment.py experiment=learned_matches data.image_list=configs/evaluation_images.txt matching.pairing=hybrid reconstruction.learned_confidence_checkpoint=outputs/point_confidence_v1/error_model_v1.pt run_name=new_combined_v1
```

Existing reports/runs are immutable. For a fresh complete run under a new root, use `scripts/run_research_suite.py --output outputs/reproduction_v1`. It generates a fresh classical baseline for confidence test tracks, runs the reference audit before supervision, and selects the filter by spatial validation. Add the optional LightGlue dependencies before running the suite.

Key new files are `geometry/reference.py`, `data/spatial.py`, `features/store.py`, `retrieval/pairs.py`, `matching/learned_filter.py`, `matching/lightglue_adapter.py`, `pointcloud/learned_confidence.py`, `evaluation/uncertainty.py` and `research_pipeline.py`. The old runner dispatches the new method through `experiment=learned_matches`, preserving the classical/old-visibility paths.

`matching.evaluation_mode=true` rejects trained/validation images and cameras outside the test slab. Set it to false only for an explicitly labeled production reconstruction; overlap with training images is recorded and those results are not spatial-holdout evaluation. Learned filtering currently requires the training preprocessing: 960px, 4,096 requested SIFT features and no CLAHE.

Local validation passes **35 pytest cases**, including reference-F/triangulation geometry, ambiguous-label exclusion, attention permutation equivariance, holdout isolation, spatial gaps, tie-aware AUSE, unique assignments and retrieval pair validity. Ruff and Black check the updated code. Raw artifacts remain under the new versioned output directories; six report figures live under `docs/figures_v1`.

## Research priorities

The immediate priority is improving C's training support and error-distribution robustness, followed by multiple spatial point blocks and uncertainty calibration. A needs evaluation on additional unseen scene regions and fair end-to-end comparison with LightGlue. B needs overlap labels supported by a scene model or observed shared tracks, followed by a budget-matched hybrid evaluation. Repeated seeds and the full-scene reconstruction remain separate benchmark work.

The new supervision is independent of this project's RANSAC decision rule. Establishing independent measured geometric truth remains an additional requirement. The evidence supports a more testable learned pipeline and increased sparse support; it does not yet support a claim of universally better reconstruction or calibrated 3D confidence.

## Primary methodological references

- [OANet paper](https://arxiv.org/abs/1908.04964): contextual correspondence classification motivates the custom pair-context architecture.
- [Official LightGlue implementation](https://github.com/cvg/LightGlue): pinned pretrained SIFT comparison baseline.
- [Ilg et al., uncertainty evaluation](https://arxiv.org/abs/1802.07095): sparsification/oracle comparison motivates the explicitly specified AUSE implementation.
"""
    summary_path.write_text(body, encoding="utf-8")
    report_path.write_text(
        body.replace("# Project summary v1", "# Research report v1", 1)
        .replace("(PROJECT_SUMMARY.md)", "(../PROJECT_SUMMARY.md)")
        .replace("(docs/figures_v1/", "(figures_v1/"),
        encoding="utf-8",
    )
    status_path.write_text(
        "# Project status v1\n\nReference gate, learned match filtering, full-dataset DINO retrieval and learned sparse confidence are implemented and evaluated.\n\n"
        "- Reference audit: 0.2934px median, 94.14% below 1.5px.\n"
        "- Learned context: AP 0.9459 / ROC AUC 0.9722 on the spatial development holdout.\n"
        "- Combined SfM: 16 cameras / 1,912 sparse points / 0.4474px reprojection error.\n"
        "- Retrieval: 1,244 images; DINO recall 9.04% vs sequential 10.08% on the default overlap proxy.\n"
        "- Learned 3D AUSE: 0.4910 vs heuristic 0.4194; uncertainty calibration remains unsuccessful.\n"
        "- Local tests: 35 passed.\n\nRead [PROJECT_SUMMARY_v1.md](PROJECT_SUMMARY_v1.md) for methods, before/after tables, limitations and commands. Original summaries/results are preserved.\n",
        encoding="utf-8",
    )
    print("Wrote versioned summaries, report, comparison JSON and six figures", flush=True)


if __name__ == "__main__":
    main()
