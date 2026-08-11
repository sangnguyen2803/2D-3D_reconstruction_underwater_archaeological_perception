"""English v3 documentation and figures generated only from measured artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matchability.report import BACKBONES, BASELINES, arrays
from underwater_vision.utils.io import read_rgb, write_json

LIMITATIONS = """Labels describe where RootSIFT top-1 matching meets reference geometry on one scene. They are independent of this project's RANSAC, but the published cameras were obtained by bundle adjustment on these images, not independent surveying. Epipolar compatibility is necessary, not proof of a correct correspondence. Non-overlap gating also uses epipolar evidence and can admit accidental alignments. Camera-center exclusion bands do not guarantee disjoint visible surfaces. Dense outputs have sparse supervision only at detected keypoints; probabilities between keypoints, on unseen scenes, for another detector or under another pair graph are unvalidated. Co-located SIFT keypoints with different scale/orientation can have different labels despite sharing one map value. Brown calibration does not model underwater refraction. No semantic classes, SAM/Grounding DINO labels, manual annotations or v2 semantic results enter these experiments. Four fixed fine-tuning epochs are an initial budget, not evidence that all architectures converged. Three seeds per region are not nine independent regions."""

REFERENCES = """- [MERMAID dataset conventions and source links](dataset.md).
- [ResNet18 pretrained weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [SegFormer official implementation](https://github.com/NVlabs/SegFormer).
- [DINOv2 official implementation](https://github.com/facebookresearch/dinov2).
- [Temperature scaling, Guo et al.](https://arxiv.org/abs/1706.04599).
- [OANet](https://arxiv.org/abs/1908.04964) and [LightGlue](https://github.com/cvg/LightGlue), historical matching controls from v1.
- [Sparsification/AUSE evaluation](https://arxiv.org/abs/1802.07095).
"""


def number(value, digits=4):
    return "unmeasured" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def plot_save(figure, root, name):
    figure.tight_layout()
    figure.savefig(root / (name + "_v3.png"), dpi=170, bbox_inches="tight")
    plt.close(figure)


def figures(result, audit, root):
    root.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    protocols = ["v1_partition", "fold0", "fold1", "fold2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(protocols))
    bottom = np.zeros(len(x))
    for kind, color in [("positive", "#208e79"), ("negative", "#e78945"), ("ignore", "#aaaeb9")]:
        values = [
            audit["audits"][p + "_hybrid"]["counts"]["pos_1.5_neg_4"]["test"][kind]
            for p in protocols
        ]
        axes[0].bar(x, values, bottom=bottom, label=kind, color=color)
        bottom += values
    axes[0].set(
        xticks=x,
        xticklabels=protocols,
        ylabel="Keypoints",
        title="L1 test labels (overlapping protocols)",
    )
    axes[0].legend()
    for protocol in protocols:
        stability = audit["audits"][protocol + "_hybrid"]["stability"]
        axes[1].plot(range(9), list(stability.values()), marker="o", label=protocol)
    axes[1].set(
        xticks=range(9),
        xticklabels=[k.replace("pos_", "").replace("_neg_", "/") for k in stability],
        xlabel="positive / negative thresholds (px)",
        ylabel="Fraction of labels unchanged",
        title="Sensitivity relative to 1.5 / 4 px",
    )
    axes[1].legend()
    plot_save(fig, root, "label_audit")

    names = BACKBONES + BASELINES
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(
        names,
        [result["summary"][n]["average_precision"]["mean"] or 0 for n in names],
        xerr=[result["summary"][n]["average_precision"]["std"] or 0 for n in names],
        color=["#2378a7"] * 3 + ["#87949e"] * len(BASELINES),
    )
    ax.set(
        xlabel="Test AP, equal-weight mean ± SD across folds/seeds",
        xlim=(0, 1),
        title="E1: 3 spatial regions, 3 seeds per model",
    )
    plot_save(fig, root, "matchability_benchmark")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for fold, ax in enumerate(axes):
        for run in result["dense_runs"]:
            if run["fold"] == fold and run["seed"] == 42:
                curve = run["test_after"]["retention"]
                ax.plot(curve["keypoint_fraction"], curve["positive_recall"], label=run["model"])
        for run in result["baseline_runs"]:
            if run["fold"] == fold and run["seed"] == 42:
                for name in ["response", "dino_mlp", "tabular_boosting"]:
                    curve = run["models"][name]["retention"]
                    ax.plot(
                        curve["keypoint_fraction"],
                        curve["positive_recall"],
                        linestyle="--",
                        label=name,
                    )
        ax.set(
            title=f"Fold {fold}, seed 42",
            xlabel="Labeled keypoints retained",
            ylabel="Positives retained",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    axes[-1].legend(fontsize=7)
    plot_save(fig, root, "retention")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for fold, ax in enumerate(axes):
        run = next(
            (
                r
                for r in result["dense_runs"]
                if r["fold"] == fold and r["seed"] == 42 and r["model"] == "segformer_b0"
            ),
            None,
        )
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        if run:
            for phase in ["test_before", "test_after"]:
                bins = run[phase]["calibration"]
                ax.plot(
                    [b["predicted"] for b in bins],
                    [b["observed"] for b in bins],
                    marker="o",
                    label=f"{phase}, ECE={run[phase]['ece']:.3f}",
                )
        ax.set(
            title=f"Fold {fold}, seed 42",
            xlabel="Predicted score",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=7)
    plot_save(fig, root, "calibration")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for fold, ax in enumerate(axes):
        run = next(
            (
                r
                for r in result["criteria"]["Q4"]["curves"]
                if r["fold"] == fold and r["seed"] == 42
            ),
            None,
        )
        if run:
            for source in ["matchability", "response", "random"]:
                rows = [r for r in run["curves"] if r["source"] == source and r["pairs"] > 0]
                ax.plot(
                    [r["budget"] for r in rows],
                    [r["original_compatible_candidate_recall"] for r in rows],
                    marker="o",
                    label=source,
                )
        ax.set(
            title=f"Fold {fold}: feasible pairs only",
            xlabel="Exact keypoint budget",
            ylabel="Original compatible-candidate recall",
            ylim=(0, 1),
        )
        ax.legend(fontsize=7)
    plot_save(fig, root, "keypoint_budget")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for fold, ax in enumerate(axes):
        run = next(
            (r for r in result["confidence_runs"] if r["fold"] == fold and r["seed"] == 42), None
        )
        if run:
            for name in ["v1_gaussian_8", "robust_8", "matchability_10", "heuristic"]:
                curve = run["models"][name]["curve"]
                ax.plot(curve["fraction_removed"], curve["retained_error_normalized"], label=name)
            ax.plot(
                curve["fraction_removed"],
                curve["oracle_error_normalized"],
                linestyle=":",
                color="black",
                label="oracle",
            )
        ax.set(
            title=f"Fold {fold}, seed 42",
            xlabel="Least confident points removed",
            ylabel="Retained error / initial mean",
            ylim=(0, 2),
        )
    axes[-1].legend(fontsize=7)
    plot_save(fig, root, "sparsification")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, model, ylabel in [
        (axes[0], "paired_dense", "segformer_b0", "Dense AP − strongest baseline AP"),
        (axes[1], "paired_filters", "context", "16-input context AP − 12-input AP"),
    ]:
        for fold in range(3):
            entry = result[key].get(f"fold{fold}", {})
            entry = (
                entry.get("models", {}).get(model, {})
                if key == "paired_dense"
                else entry.get(model, {})
            )
            if "ci95" in entry:
                lo, hi = entry["ci95"]
                ax.plot([fold, fold], [lo, hi], color="#2378a7")
                ax.scatter([fold], [entry["estimate"]], color="#2378a7")
        ax.axhline(0, color="grey", linestyle="--")
        ax.set(
            xticks=range(3),
            xticklabels=["fold0", "fold1", "fold2"],
            ylabel=ylabel,
            title="Mean of 3 seeds; paired 95% cluster CI",
        )
    plot_save(fig, root, "paired_comparisons")


def heatmaps(result, dataset, root):
    """Explicit best/worst labeled-image AP, illustrative and excluded from fitting."""
    from underwater_vision.matchability.build import load_features
    from underwater_vision.matchability.infer import MatchabilityPredictor

    selected = []
    for fold in range(3):
        run = next(
            (
                r
                for r in result["dense_runs"]
                if r["fold"] == fold and r["seed"] == 42 and r["model"] == "segformer_b0"
            ),
            None,
        )
        if not run:
            continue
        ranking = []
        for file in sorted((Path(run["path"]) / "predictions").glob("*.npz")):
            r = arrays(file)
            known = r["labels"] >= 0
            if known.sum() >= 30 and len(np.unique(r["labels"][known])) == 2:
                ranking.append(
                    (
                        average_precision_score(r["labels"][known], r["probability"][known]),
                        str(r["name"]),
                    )
                )
        if not ranking:
            continue
        ranking.sort()
        predictor = MatchabilityPredictor(Path(run["path"]) / "model_v3.pt", "cpu")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for row, (score, name) in enumerate([ranking[-1], ranking[0]]):
            rgb = read_rgb(dataset / "images" / (name + ".png"))
            feature = load_features(dataset, name)
            values = predictor.predict(rgb)
            with np.load(dataset / "labels" / f"fold{fold}" / "hybrid" / (name + ".npz")) as z:
                labels = z["l1"]
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(values, vmin=0, vmax=1, cmap="viridis")
            axes[row, 2].imshow(rgb)
            for value, color in [(1, "lime"), (0, "red"), (-1, "grey")]:
                xy = feature["xy"][labels == value]
                axes[row, 2].scatter(xy[:, 0], xy[:, 1], s=3, c=color, alpha=0.6)
            axes[row, 0].set_title(
                f"{'Highest' if row == 0 else 'Lowest'} image AP={score:.3f}\n{name}", fontsize=8
            )
            axes[row, 1].set_title("Dense score (CPU FP32 illustration)")
            axes[row, 2].set_title("L1: positive green / negative red / ignore grey")
            selected.append(
                {
                    "fold": fold,
                    "seed": 42,
                    "name": name,
                    "image_ap": float(score),
                    "selection": "highest image AP" if row == 0 else "lowest image AP",
                    "labeled_keypoints": int(np.sum(labels >= 0)),
                }
            )
            for ax in axes[row]:
                ax.axis("off")
        plot_save(fig, root, f"heatmaps_fold{fold}")
    write_json(root / "heatmap_selection_v3.json", selected)


def dataset_card(manifest, audit, l2):
    rows, pairs, sensitivity, stronger = [], [], [], []
    for protocol in manifest["protocols"]:
        name = protocol["name"]
        a = audit["audits"][name + "_hybrid"]
        for slab in ["train", "validation", "test"]:
            counts = a["counts"]["pos_1.5_neg_4"][slab]
            rows.append(
                [
                    name,
                    slab,
                    len(protocol["groups"][slab]),
                    *[counts[k] for k in ["positive", "negative", "ignore"]],
                ]
            )
            p = a["pairs"][slab]
            eligible = p["eligible_by_positive_threshold"][1]
            pairs.append(
                [
                    name,
                    slab,
                    p["candidate_pairs"],
                    eligible,
                    number(1 - eligible / max(1, p["candidate_pairs"]), 3),
                ]
            )
            if name in l2["protocols"]:
                stronger.append(
                    [
                        name,
                        slab,
                        *[
                            l2["protocols"][name]["counts"][slab][k]
                            for k in ["positive", "negative", "ignore"]
                        ],
                    ]
                )
        for threshold, stability in a["stability"].items():
            sensitivity.append([name, threshold, number(stability)])
    return f"""# Matchability label dataset card v3

## Source and inputs

MERMAID EPITA underwater archaeology acquisition: 1,244 images with published camera poses (1,272 source images in total). All 1,244 posed images are processed locally; reference-free images are outside the label benchmark. RGB is resized to 960×720, saved losslessly as PNG, without CLAHE. RootSIFT requests 4,096 features; actual detections are 671,722 in total, minimum 1, median 218, maximum 4,097. Requesting features does not ensure an exact count. Exact-budget experiments skip infeasible pairs and report availability.

Source manifest SHA256: `{digest(Path('outputs/matchability_labels_v3/manifest_v3.json'))}`. Source images, camera calibration and pose conventions remain as recorded in the v1 dataset/reference audit. The image dataset is supplied under CC BY-NC-ND 4.0; this repository's MIT license applies to its code, not those assets. Do not publish image derivatives or trained checkpoints assuming code licensing covers them.

## Labels and graph

For each within-slab sequential-window-6 or DINO-retrieved pair, compute bidirectional top-1 RootSIFT candidates and sqrt-Sampson error using Brown-undistorted coordinates and reference F. A hybrid pair is eligible when at least 15 directed candidates have error ≤1.5 px. Mutual candidates count in both directions; the audit explicitly uses this directed count. Each keypoint is positive if any eligible candidate is ≤1.5 px; negative if at least two eligible candidates exist and all are ≥4 px; otherwise ignore. Ignored keypoints enter neither loss nor evaluation. Soft targets use good/(good+bad), excluding the ambiguous band. L2 certifies L1 positives using conflict-free mutual reference tracks and median error over three held-out-view triangulations ≤2 px; L1 negatives remain negative and uncertified positives become ignore. L2 and soft experiments use common L1 validation/test labels.

## Spatial slabs and L1 counts (hybrid graph)

{table(['Protocol','Slab','Images','Positive','Negative','Ignore'], rows)}

The v1 partition excludes 210 images. Rotated protocols exclude 155 and rotate three fixed physical blocks (377, 336, 376 views), with the v1 0.4471966-unit camera-center gap. Groups are disjoint within each protocol. These protocols overlap in their source images and are not four independent datasets.

## Pair eligibility at frozen 1.5 px threshold

{table(['Protocol','Slab','Candidate pairs','Eligible pairs','Removed fraction'], pairs)}

Eligibility is computed separately for each positive threshold. The full immutable E0 JSON contains both sequential and hybrid graphs, all per-slab counts and reference-error histograms. These directed top-1 errors include non-overlap candidates and should not be interpreted as a histogram of verified inliers.

## Threshold sensitivity: fraction of labels unchanged versus 1.5/4 px

{table(['Protocol','Positive/negative setting','Agreement'], sensitivity)}

The nine settings are positive {{1, 1.5, 2}} px × negative {{3, 4, 6}} px. Proposed thresholds, minimum counts and four-epoch budget were retained and frozen after E0, before fitting final models. Agreement includes ignore labels; it is not a measure of annotation truth.

## Stronger L2 counts

{table(['Protocol','Slab','Positive','Negative','Ignore'], stronger)}

## Limitations

{LIMITATIONS}
"""


def build_documents(comparison, tests_path=None, include_heatmaps=True):
    result = read_json(comparison)
    dataset = Path(result["dataset"])
    manifest, audit = read_json(dataset / "manifest_v3.json"), read_json(dataset / "audit_v3.json")
    l2 = read_json("outputs/matchability_l2_labels_v3/audit_v3.json")
    fig_root = Path("docs/figures_v3")
    figures(result, audit, fig_root)
    if include_heatmaps:
        heatmaps(result, dataset, fig_root)
    benchmark = []
    for name in BACKBONES + BASELINES:
        s = result["summary"][name]
        benchmark.append(
            [
                name,
                s["average_precision"]["n"],
                number(s["average_precision"]["mean"]),
                number(s["average_precision"]["std"]),
                number(s["roc_auc"]["mean"]),
            ]
        )
    dense_pairs, filter_pairs = [], []
    for fold, r in result["paired_dense"].items():
        for name, c in r["models"].items():
            dense_pairs.append(
                [
                    fold,
                    name,
                    r["strongest_baseline"],
                    number(c.get("estimate")),
                    str(c.get("ci95", c.get("status"))),
                ]
            )
    for fold, r in result["paired_filters"].items():
        for name, c in r.items():
            filter_pairs.append(
                [fold, name, number(c.get("estimate")), str(c.get("ci95", c.get("status")))]
            )
    criteria = table(
        ["Criterion", "Measured decision"],
        [[k, v["status"]] for k, v in result["criteria"].items()],
    )
    calibration = table(
        ["Fold", "Seed", "Temperature", "ECE before", "ECE after", "Brier before", "Brier after"],
        [
            [
                r["fold"],
                r["seed"],
                number(r["temperature"]),
                number(r["test_before"]["ece"]),
                number(r["test_after"]["ece"]),
                number(r["test_before"]["brier"]),
                number(r["test_after"]["brier"]),
            ]
            for r in result["dense_runs"]
            if r["model"] == "segformer_b0"
        ],
    )
    filters = table(
        ["Filter", "Runs", "AP mean", "AP SD", "ROC AUC", "F1"],
        [
            [
                n,
                result["summary"][n]["average_precision"]["n"],
                *[number(result["summary"][n]["average_precision"][k]) for k in ["mean", "std"]],
                number(result["summary"][n]["roc_auc"]["mean"]),
                number(result["summary"][n]["f1"]["mean"]),
            ]
            for n in ["context_12", "context_16", "mlp_12", "mlp_16"]
        ],
    )
    points = table(
        [
            "Point model",
            "AUSE mean",
            "AUSE SD",
            "Spearman(error)",
            "MAE px",
            "90% interval coverage",
        ],
        [
            [
                n,
                *[number(result["summary"][n]["ause"][k]) for k in ["mean", "std"]],
                number(result["summary"][n]["spearman_error"]["mean"]),
                number(result["summary"][n]["prediction_mae_px"]["mean"]),
                number(result["summary"][n]["interval_coverage90"]["mean"]),
            ]
            for n in ["v1_gaussian_8", "robust_8", "matchability_10", "heuristic", "view_count"]
        ],
    )
    deployment = result.get("deployment", {})
    timing = table(
        [
            "Hardware",
            "Precision",
            "Batch",
            "Mean ms",
            "p95 ms",
            "Images/s",
            "Peak allocated VRAM MiB",
        ],
        [
            [
                r["device"],
                r["precision"],
                r["batch"],
                number(r["latency_mean_ms"], 2),
                number(r["latency_p95_ms"], 2),
                number(r["throughput_images_per_second"], 2),
                (
                    number(r["peak_vram_bytes"] / 2**20, 1)
                    if r["peak_vram_bytes"] is not None
                    else "n/a"
                ),
            ]
            for r in deployment.get("timing", [])
        ],
    )
    sfm = result.get("sfm", {})
    sfm_table = table(
        ["Method", "Seed", "Status", "Cameras", "Points", "Reprojection px", "ATE", "Failure"],
        [
            [
                r["method"],
                r["seed"],
                r["status"],
                r.get("registered_images", "—"),
                r.get("sparse_points", "—"),
                number(r.get("mean_reprojection_error_px")),
                number(r.get("ate_rmse")),
                r.get("error", "—").replace("|", "/"),
            ]
            for r in sfm.get("reconstructions", [])
        ],
    )
    labels_table = table(
        ["Labels", "Seed", "Common L1 test AP", "ROC AUC", "ECE"],
        [
            [
                r["variant"],
                r["seed"],
                number(r["test"]["average_precision"]),
                number(r["test"]["roc_auc"]),
                number(r["test"]["ece"]),
            ]
            for r in sfm.get("label_ablation", [])
        ],
    )
    verification = read_json(tests_path) if tests_path else {"status": "unrecorded"}
    common = f"""## Measured pre-registered decisions

{criteria}

Q1 uses the primary SegFormer model against the strongest measured baseline in each fold, including both tabular controls and the retrained DINO-MLP. CIs are conditional on that baseline selection. Q2 requires every seed of each fold to satisfy ECE ≤0.05; a passing mean does not conceal a failing seed. Q3 compares retrained context filters on identical candidates. Q4 follows frozen compatible-candidate recall and reports infeasible budgets explicitly. Q5 requires the ten-feature learner to outperform both the retrained v1 learner and the heuristic under paired AUSE intervals. Failed criteria do not support an improvement claim.

## E1: dense matchability and controls

{table(['Model','Runs','AP mean','AP SD','ROC AUC mean'], benchmark)}

Means and SD summarize the nine fold/seed runs equally. Only three spatial test regions are independent at the region level. Paired image-bootstrap intervals use 500 resamples with exact weighted AP and score ties pooled; the same cluster draws are aligned across seeds before averaging the model differences.

{table(['Fold','Dense model','Strongest baseline','AP difference','95% CI'], dense_pairs)}

![E1 measured benchmark](figures_v3/matchability_benchmark_v3.png)

![Retention at labeled keypoints](figures_v3/retention_v3.png)

## E2: stronger and soft supervision

{labels_table}

All variants use identical v1 camera-center slabs, four epochs, three seeds and the original common L1 validation/test labels. The 743-view test block is the historical v1 partition; it is not an additional independent scene. L2 positives are fewer, and all unconfirmed positives are ignored rather than flipped to negatives.

## E3: calibration on held-out regions

{calibration}

Temperature is fitted on validation logits only and stored per checkpoint. It preserves ranking. ECE uses ten equal-width score bins; the raw per-fold calibration bins and Brier values are retained. A map can have useful ranking while its numeric values fail calibration on another region. Use the term matchability score unless Q2 passes.

![Reliability diagrams](figures_v3/calibration_v3.png)

## E4: correspondence filtering

{filters}

{table(['Fold','Architecture','16-minus-12 AP','95% CI'], filter_pairs)}

Both twelve- and sixteen-input filters train on the same 64 selected train views and 32 validation views with nearest-center and distant training pairs. Rotated-fold test pools use 32 evenly sampled views across each full slab, with a six-view window over that selected list. This deliberately includes difficult/non-overlapping pairs, unlike the compact historical sixteen-view v1 development sequence. Ambiguous reference-F matches are ignored. Test pair eligibility is not restricted to the dense-label graph: the filter must reject bad candidates from non-overlap as well. Lower AP than historical v1 is not a before/after decline on identical inputs. The causal effect of the four new inputs is the paired twelve-versus-sixteen comparison here. Precision/recall/F1 thresholds are selected on validation only.

![Paired AP differences](figures_v3/paired_comparisons_v3.png)

## E5: exact keypoint budgets

Budgets are 512/1,024/2,048/4,096, with identical RootSIFT pools and exactly N keypoints at both endpoints. SIFT often detects fewer than requested: the median is 218. Infeasible pairs are recorded, not silently clipped or replaced. Curves therefore describe different feasible pair subsets across N; they are not a fully observed end-to-end speed/quality frontier. Original-candidate recall counts full-pool reference-compatible candidates whose two indices survive selection; rematching produces a separate compatible count and precision. Random seeds are independent per image using SHA256(seed:image name), reused consistently across its pairs. Matching timing includes descriptor transfer and synchronization, with five warm-up calls per budget/source, but excludes extraction and map inference.

![Budget recall on feasible pairs](figures_v3/keypoint_budget_v3.png)

## E6: identical sixteen-view SfM inputs

{sfm_table}

The same sixteen image names are used for classical, retrained twelve-input context, matchability context and exact top-512. Every configuration is attempted at all three seeds. These are development images already examined in v1, not fresh test evidence. A missing exact budget is a recorded failed configuration; it does not establish a reconstruction advantage. The historical frozen v1 context is separately rerun with matchability off: 16 cameras, 1,890 points, mean reprojection error 0.4504205128199967 px, matching the preserved 0.4504205128199975 px within 1e-12. This verifies numerical backward compatibility, not an improved method.

## E7: point-error prediction

{points}

Reference-compatible tracks use all available non-excluded views of each slab. One or two observations are held out in each of three deterministic repeat groups; all target-view locations, DINO, reliability and matchability values are excluded from input features. Reprojection error is a reference-camera proxy. Ten inputs are the original eight plus mean/min matchability over fitting views. Robust eight-input controls isolate the feature effect from the changed loss and conformal calibration. Track-median predictions are used for evaluation; paired bootstrap keeps each track intact. Ninety-percent intervals use a separate validation-track calibration subset in log1p(error). Nominal coverage is not guaranteed under region/domain shift. Heuristic and view-count scores have no pixel-error units, so their prediction MAE is not reported. AUSE uses the original v1 mean-normalized 100-step definition with tie averaging; lower is better. Positive Spearman means larger predicted error ranks larger true error.

![Point sparsification](figures_v3/sparsification_v3.png)

## Error analysis

`outputs/comparison_v3/binned_analysis_v3.json` includes E1, E4 and E7 at seed 42 in every fold, binned by local contrast, SIFT scale, image sharpness, brightness and frame radius. Bins are descriptive evaluation quartiles, not tuned thresholds. Heatmaps show the highest and lowest per-image test AP with at least 30 known keypoints and both classes, for each fold at seed 42. Their selection is saved in `docs/figures_v3/heatmap_selection_v3.json`; these chosen extremes illustrate failures and are not unbiased performance samples. Dense heatmaps are CPU FP32 illustrations; benchmark logits use CUDA AMP FP16.

## E8: export and service

Trained checkpoint SHA256: `{deployment.get('checkpoint_hash', 'unmeasured')}`. ONNX SHA256: `{deployment.get('onnx_sha256', 'unmeasured')}`. Dynamic batch with fixed 720×960 input is tested. ONNX/CPU FP32 max absolute logit difference: {number(deployment.get('max_absolute_logit_error'), 8)}, tolerance atol=1e-4, rtol=1e-3.

{timing}

Timing is PyTorch model-only after five warm-up passes and twenty repeats, NCHW RGB [0,1], batch one, 720×960. It excludes image decoding, resizing, transfers, keypoint sampling and HTTP. Peak VRAM is PyTorch allocated memory, not total driver/process memory. FP16 differences from CPU FP32 are reported separately. Measurements apply only to this local hardware and run conditions.

The CPU ONNX FastAPI Docker image was actually built, launched, and tested on a real image: a 720×960 16-bit PNG map, top-32 keypoints with original indices, `/health`, and invalid-image HTTP 400. Evidence and the exact image/container IDs are in `outputs/matchability_docker_smoke_v3`. The container/engine was stopped afterward to release resources. The single HTTP smoke duration includes all service work and is not the latency benchmark above.

## Verification and preserved negative runs

Verification artifact: `{tests_path or 'unrecorded'}`; status `{verification.get('status')}`. Old v1/v2 reports, summaries and results remain immutable. New run directories refuse to overwrite an existing name. Configs, seeds, split manifests, label/audit hashes and model hashes are saved. Cache keys include image state, resolution, checkpoint, temperature, backend/device/precision. Frozen DINO evaluation grids are losslessly cached as FP32 with original strides, bounded to 5 GB; augmented training crops never use this cache. The initial BF16 grid-cache audit failed (max logit error 0.0019002) and is retained. The replacement cache passed with zero grid/logit difference on the audited real image. The early DINO fold0 seed43 run is retained and superseded by its corrected four-epoch rerun.

## Scientific limitations

{LIMITATIONS}
"""
    write_new("docs/dataset_matchability_v3.md", dataset_card(manifest, audit, l2))
    write_new(
        "docs/research_report_v3.md",
        "# Dense matchability reconstruction research report v3\n\n"
        + common
        + "\n## References\n\n"
        + REFERENCES,
    )
    modelcard = f"""# Dense matchability model card v3

## Intended use

Experimental matchability scoring for RGB images from the MERMAID EPITA scene at 960×720, paired with the exact RootSIFT/reference-label and spatial-fold protocols. One-channel output is sparsely supervised at keypoints, not a semantic segmentation mask. The primary deployment checkpoint is SegFormer-B0 pretrained from NVIDIA MIT-B0, revision `80983a413c30d36a39c20203974ae7807835e2b4`, then fine-tuned for four fixed epochs. Checkpoint selection uses spatial validation AP; scalar temperature uses validation labels only. The deployment fold0 seed42 checkpoint is a reproducible exemplar, not selected as the strongest test seed.

## Input and output

NCHW float RGB [0,1], normalization embedded. ONNX accepts dynamic batch and fixed 720×960 resolution. Output is a one-channel logit map; apply sigmoid(logit/temperature), then bilinearly sample original OpenCV keypoint coordinates. The service resizes bounded RGB, returns a 16-bit PNG score map, summary statistics and optional top-N RootSIFT keypoints. Exact N is rejected when too few features exist. On arbitrary aspect ratios/resolutions, service preprocessing differs from the benchmark and is unvalidated.

## Measured decisions and performance

{criteria}

{table(['Model','Runs','AP mean','AP SD','ROC AUC mean'], benchmark[:3])}

{table(['Fold','Dense model','Strongest baseline','AP difference','95% CI'], [r for r in dense_pairs if r[1] == 'segformer_b0'])}

{calibration}

{timing}

This card describes three independent spatial blocks of one scene. A useful ranking does not establish calibrated probabilities when Q2 fails. Refer to the research report for negative/null downstream effects and complete baseline metrics.

## Licenses

Project code: MIT. [Torchvision code](https://github.com/pytorch/vision/blob/main/LICENSE): BSD-3-Clause; ImageNet-derived pretraining has separate data terms. [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE): Apache-2.0. [NVIDIA MIT-B0 model card](https://huggingface.co/nvidia/mit-b0) points to the [SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE), which restricts the work and derivatives to noncommercial research/evaluation; public availability does not imply commercial permission. MERMAID imagery: CC BY-NC-ND 4.0 as recorded in the dataset card. A project MIT license does not supersede dataset or pretrained-weight terms. Check the applicable terms before redistributing any trained checkpoint or derived imagery; no checkpoint publication is performed here.

## Limitations and failure modes

{LIMITATIONS}

Spatial calibration shifts, texture-poor frames, blur/backscatter, repeated surfaces, image borders, tiny feature pools and low-overlap pairs can degrade behavior. Sparse keypoint supervision does not validate every pixel. Reliability replacement is detachable and remains experimental; passing API/parity checks proves implementation behavior, not scientific benefit.

## Artifacts

Checkpoint: `{deployment.get('checkpoint_hash', 'unmeasured')}`. ONNX: `{deployment.get('onnx_sha256', 'unmeasured')}`. [Research report](research_report_v3.md), [label dataset card](dataset_matchability_v3.md), [comparison JSON](../outputs/comparison_v3/comparison_v3.json), [verification](../outputs/verification_v3/verification_v3.json).
"""
    write_new("docs/model_card_matchability_v3.md", modelcard)
    modules = """| Module | Responsibility | Detachment |
| --- | --- | --- |
| `matchability/labels.py`, `build.py`, `strong_labels.py` | Independent reference-F L1/soft labels, stronger track labels, eligibility and audit | Standalone dataset-builder CLIs |
| `matchability/model.py`, `train.py`, `dataset.py`, `loss.py` | Three pretrained dense encoders, sparse BCE, coordinate-preserving crops, seeded AMP training | Standalone checkpoints; no reconstruction imports needed for inference |
| `matchability/infer.py`, `calibration.py` | Full map, exact keypoint sampling, hashes/cache and temperature | Use `MatchabilityPredictor` or ONNX predictor alone |
| `matchability/tabular_baseline.py` | Random/response/scale, handcrafted, DINO-MLP, logistic/boosting controls | Same labels, separate baseline artifacts |
| `matchability/experiments.py` | Retrained 12/16-input filters and budget curves | `matching.matchability_features="off"` restores original filter interface |
| `visibility/source.py` | Explicit handcrafted, DINO-head or matchability reliability | Source chosen independently of the four filter features |
| `matchability/confidence.py` | Robust 8/10-input point-error learning, held-out targets, conformal intervals | Omit `reconstruction.matchability_confidence_checkpoint` to disable |
| `evaluation/matchability.py`, `matchability/report.py` | Ignore-aware metrics, exact paired AP bootstrap, bins, immutable comparison | Read-only aggregation of saved runs |
| `serving/matchability_api.py`, `matchability_export.py` | Trained ONNX parity/timing and FastAPI/Docker | `--task matchability`; v2 semantic task remains separate |
"""
    summary = f"""# Project summary v3 — visibility-aware underwater 2D–3D reconstruction

## Current state

The project now includes a trained dense matchability component supervised by published reference geometry, three pretrained backbone comparisons, seven controls, independent spatial evaluation, four detachable integration points, and a tested ONNX/FastAPI Docker service. This extends the original image-to-SfM/MVS project. The v2 semantic prototype remains experimental and is excluded from v3 evidence. Completion state: `{result['status']}`. All numerical claims below come from saved local experiments, not estimates.

{criteria}

## Original and new inputs/outputs

Original input: underwater RGB image sequences, camera calibration and optional reference poses for evaluation. Original outputs: RootSIFT/DINO correspondences, RANSAC/COLMAP sparse cameras and points, optional dense MVS/cloud/mesh, heuristic and learned confidence, trajectory/reprojection metrics and plots. Reference cameras provide supervision/evaluation and never select COLMAP inliers at deployment.

New training input: the same lossless RGB960×720 images, original OpenCV RootSIFT coordinates, L1/L2/soft labels from reference geometry and camera-center spatial manifests. No manual annotation or semantic classes. New network output: an H×W logit/score map and sampled per-keypoint scores. New integration outputs: sixteen-input match probabilities, exact-budget selected keypoint indices, alternative reliability, ten-input point-error estimates/intervals, ONNX and HTTP score-map responses.

## Data already available

The local MERMAID dataset contains 1,272 images, of which 1,244 have reference poses; no additional download is required. All 1,244 posed images have v3 labels. The v1 partition is 226 train / 65 validation / 743 test / 210 excluded. Three rotated protocols use fixed 377/336/376-view blocks plus 155 excluded images and a 0.4471966-unit gap. Feature detections total 671,722 (median 218), despite 4,096 requested; this limits exact top-N comparisons.

## Architecture and modularity

{modules}

The normal pipeline remains available when optional fields are absent. The original twelve-input context checkpoint is rejected if sixteen inputs are requested; a sixteen-input checkpoint also rejects another matchability checkpoint hash. Class-based v2 feature inputs cannot be mixed silently with matchability inputs. Fit/evaluation names and out-of-slab views are checked before reconstruction. The matchability-off rerun reproduces the original 1,890-point result to machine precision.

## Training and evaluation protocol

All three encoders start pretrained: ResNet18-UNet, SegFormer-B0 and frozen DINOv2 B/14 with a conv head. Sparse BCE samples original coordinates with bilinear grid sampling, excludes ignore, and uses class weighting. Training uses 384×512 crops, photometric changes, AdamW (encoder 2e-5/head 3e-4), AMP FP16, cosine schedule, gradient clipping, batch one and accumulation four. Three seeds (42/43/44), four fixed epochs, spatial validation-AP checkpoint selection, and validation-only scalar temperature are logged with MLflow. DINO's encoder is frozen; its evaluation grids are cached losslessly, while augmented crops are recomputed. These are controlled initial-budget results rather than a claim of convergence.

## Principal measured results

{table(['Model','Runs','AP mean','AP SD','ROC AUC mean'], benchmark)}

{filters}

{points}

Mean/SD spans folds and seeds; the number of independent regions is three. Criterion decisions determine whether these numbers support an improvement claim. Detailed paired confidence intervals, calibration, failures and bins are in the linked research report and comparison JSON.

## Comparison with earlier versions

V1 historical context AP was 0.945910 on a compact development candidate pool; its point learner AUSE was 0.491018 versus heuristic 0.419388 with a much smaller training-track set. Rotated-fold v3 populations differ, so these numbers cannot be subtracted to claim improvement. The fair v3 comparisons retrain twelve-input match and eight-input point baselines on exactly the new populations. Robust eight-input point controls also isolate the effect of added matchability from the change in loss/calibration.

On the original sixteen development views, matchability off yields 16 cameras/1,890 points/0.4504205128199967 px, matching the preserved v1 output within 1e-12. The measured E6 table below separately compares newly retrained methods on those identical inputs.

{sfm_table}

## Deployment and verification

{timing}

Actual trained ONNX parity max absolute logit error: {number(deployment.get('max_absolute_logit_error'), 8)}; dynamic batch passed. FastAPI accepts an image, returns a 16-bit score PNG, statistics and optional top-N keypoints. Docker was built and launched; a real image and invalid upload were tested. This service smoke time is distinct from model-only timing. Verification status `{verification.get('status')}`; see the saved verification artifact for pytest/Ruff/Black output. No Git commit/push is performed without the user's per-commit review workflow.

## Run and disable components

```powershell
# Inspect a fresh immutable plan; use a different name for the actual suite.
.venv\\Scripts\\python.exe scripts/run_matchability_ablation.py --output outputs/reproduce_plan_v3 --plan-only
# Build labels and run E0–E8 under a new root (downloads cached pretrained weights as needed).
.venv\\Scripts\\python.exe scripts/run_matchability_ablation.py --output outputs/reproduce_suite_v3
# Export the trained model and start a standalone service.
.venv\\Scripts\\python.exe scripts/export_onnx.py --task matchability --checkpoint outputs/matchability_suite_v3/segformer_b0_fold0_s42_v3/model_v3.pt --output outputs/reexport_v3
.venv\\Scripts\\python.exe scripts/serve_api.py --task matchability --onnx outputs/matchability_suite_v3/deployment_v3/matchability_v3.onnx --metadata outputs/matchability_suite_v3/deployment_v3/deployment_v3.json
```

For a reconstruction with matchability inputs, use `experiment=matchability_matches` with matched dense/filter checkpoints from the same spatial protocol. For standalone reliability, choose `visibility.source` and its checkpoint independently. Restore original behavior by selecting the original experiment/checkpoint, omitting optional visibility and confidence fields, setting `matching.matchability_features="off"`, and removing keypoint-budget fields. Exact-budget mode fails clearly if too few features exist. See [usage](docs/matchability_usage_v3.md) for complete commands.

## Artifacts and scope limits

- [Detailed research report](docs/research_report_v3.md): E0–E8, all criteria, null results, CIs and calibration.
- [Label dataset card](docs/dataset_matchability_v3.md) and [model card](docs/model_card_matchability_v3.md).
- [Central measured comparison](outputs/comparison_v3/comparison_v3.json) and [binned analysis](outputs/comparison_v3/binned_analysis_v3.json).
- [Verification](outputs/verification_v3/verification_v3.json), `docs/figures_v3`, and immutable run subdirectories in `outputs/matchability_suite_v3`.

{LIMITATIONS}

Existing v1/v2 reports/results remain unchanged. Dense MVS changes, semantic labeling, enhancement E9, new-scene validation and full 1,244-view end-to-end COLMAP/MVS are outside this v3 measurement. All 1,244 images were labeled; this is distinct from reconstructing the entire sequence.
"""
    write_new("PROJECT_SUMMARY_V3.md", summary)
    write_new(
        "PROJECT_STATUS_v3.md",
        f"# Project status v3\n\nArtifact state: `{result['status']}`.\n\n{criteria}\n\nVerification: `{verification.get('status')}`.\n\nSee [summary](PROJECT_SUMMARY_V3.md) and [research report](docs/research_report_v3.md) for measured limits and preserved failures.\n",
    )
    write_json(
        fig_root / "provenance_v3.json",
        {
            "comparison_sha256": digest(comparison),
            "verification": verification,
            "figure_scope": "Seed-42 curves per region and all-seed benchmark; examples explicitly selected best/worst image AP",
        },
    )
    return result
