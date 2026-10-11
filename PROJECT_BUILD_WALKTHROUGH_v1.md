# Project Build Walkthrough v1

**Project:** Visibility-Aware Underwater 2D–3D Reconstruction  
**Snapshot:** 9 October 2026  
**Scope:** baseline, reference-supervised v1, dense-matchability v3, and subsequent reconstruction experiments.

This document explains how the project was assembled, why each component was added, how data moves between components, and what has actually been demonstrated. It is a technical walkthrough of the development stages, not a timestamped record of every edit or commit. V2 is omitted from the retained research narrative. The document is new; previous summaries and results remain preserved.

## 1. Define the problem and separate the evaluation tasks

The starting problem was to reconstruct an underwater archaeological scene from overlapping RGB photographs. Underwater imagery introduces low contrast, blur, uneven illumination, color imbalance and weak texture. These conditions can reduce the number of usable image correspondences and make geometric estimation unstable.

The system was therefore built around two connected questions:

1. Can the images produce a usable classical reconstruction?
2. Can learned reliability improve correspondence selection or downstream geometry beyond that classical reference?

These questions require different measurements. Match classification, probability calibration, camera-pose accuracy and dense-surface accuracy are separate tasks. A higher classification AP does not automatically imply a lower reconstruction error.

The intended inputs and outputs were made explicit before adding learned models:

| Stage | Input | Output |
|---|---|---|
| Data preparation | RGB photographs, calibration and image selection | Prepared RGB images and input manifest |
| Feature extraction | Prepared RGB | Keypoint coordinates, RootSIFT descriptors, optional DINOv2 descriptors |
| Pair selection | Image order or global descriptors | List of image pairs to match |
| Matching | Features from two images | Candidate correspondences and scores |
| Verification | Correspondences and camera model | Geometrically verified matches |
| Sparse SfM | Verified matches and images | Estimated cameras, tracks and sparse RGB points |
| Optional dense MVS | Registered cameras and images | Dense point cloud and mesh |
| Evaluation | Predictions and eligible reference evidence | Metrics, figures and reports |

Reference camera poses are used for supervision and evaluation. In the controlled reconstruction benchmarks, they do not initialize camera estimates or select test matches.

## 2. Establish the package and experiment structure

The implementation was organized as an installable Python package with a `src` layout. Reusable algorithms live in `src/underwater_vision`; command-line experiment entry points live in `scripts`; Hydra configurations live in `configs`.

The main responsibilities are:

```text
src/underwater_vision/
  data/             dataset loading and spatial partitions
  preprocessing/    image preparation
  features/         RootSIFT, frozen DINOv2 and feature caching
  retrieval/        cross-view pair retrieval
  visibility/       local reliability and cross-view consistency
  matching/         classical matching and learned filters
  geometry/         geometric verification and reference geometry
  reconstruction/   COLMAP adapters and reconstruction stages
  pointcloud/       point confidence and mesh cleanup
  matchability/     dense score-map learning, inference and calibration
  evaluation/       classification, trajectory and uncertainty metrics
  visualization/    figures and reconstruction inspection
  serving/          inference API and deployment support
```

The environment combines Python, NumPy, SciPy, OpenCV, PyTorch, Hydra, PyCOLMAP and Matplotlib. Optional dependency groups cover DINOv2 learning/inference, MLflow tracking, Open3D/Plotly visualization, research baselines, SegFormer, ONNX and FastAPI.

Repository setup includes packaging, dependency declarations, environment records, code licensing, tests and formatting/lint configuration. Dataset and pretrained-model licensing remain separate from the repository's code license.

This separation matters operationally: downloading the data, training a model and reconstructing a scene are different commands. A fresh Git clone does not automatically include the local images, checkpoints, COLMAP binaries or experimental outputs.

**Relevant files:** `pyproject.toml`, `environment.yml`, `requirements-lock.txt`, `configs/config.yaml`, `scripts/run_experiment.py`.

## 3. Download and inspect the real dataset

The project uses the **EPITA/LRE Mermaid underwater archaeology acquisition**. The verified local dataset contains **1,244 RGB photographs and 1,244 published reference camera poses**.

The downloader was implemented to discover official assets, resume or retry transfers, validate expected sizes, record local SHA-256 hashes and check ZIP CRCs. Extraction also protects against paths escaping the destination directory.

Inspection then establishes the actual local image count, dimensions and available metadata. This is necessary because descriptions of the dataset do not all agree on image count and resolution.

The verified source images are **3,840 × 2,880**. The standard reported comparison resolution is **960 × 720**.

Two important distinctions were recorded:

- The reference camera poses come from published bundle adjustment, rather than an independent camera survey.
- The reference STL describes the GCP network; it is not a surveyed dense surface suitable for reporting mesh Chamfer distance or F-score.

**Relevant files:** `scripts/download_data.py`, `scripts/download_sample.py`, `scripts/inspect_dataset.py`, `scripts/verify_dataset.py`, `src/underwater_vision/data/dataset_loader.py`.

## 4. Prepare images and make experiment inputs reproducible

The image-preparation stage loads RGB consistently, resizes images and keeps the camera model consistent with the image dimensions. Optional enhancement is configurable, but the reported learned comparisons use the model's expected preprocessing rather than silently changing the image distribution.

Explicit image lists allow the same acquisition IDs to be reused across baselines and learned variants. Resolved configuration files and manifests record which images and settings entered each run.

Early reconstruction pilots used small selections, including a 16-image experiment, to verify the complete pipeline before broader evaluation. Those pilots are not the full scope of learning or dataset preparation.

**Relevant files:** `scripts/preprocess.py`, `scripts/make_selection.py`, `src/underwater_vision/preprocessing/images.py`, `configs/data/mermaid.yaml`.

## 5. Build the classical local-feature baseline

The first dependable image representation is SIFT keypoint localization with RootSIFT descriptors:

1. Convert RGB to grayscale.
2. Detect SIFT keypoints and compute the 128-dimensional descriptors.
3. L1-normalize each nonnegative descriptor.
4. Take its elementwise square root.

The descriptor transformation is:

```text
RootSIFT(d) = sqrt(d / (sum(d) + epsilon))
```

This supplies precise local coordinates together with descriptors suitable for nearest-neighbor matching. The original detector uses a contrast threshold of `0.04` and a requested feature cap of `4096`.

Feature extraction is cached so that changes to scoring or reconstruction do not require recomputing identical descriptors. Cache provenance must distinguish image preparation and extractor settings. The later nondefault contrast setting enters the cache key, while the original default key remains compatible with existing caches.

**Relevant files:** `src/underwater_vision/features/local.py`, `src/underwater_vision/features/store.py`, `scripts/extract_features.py`, `configs/model/sift.yaml`.

## 6. Create and verify classical matches

For each selected image pair, local descriptor matching generates candidate correspondences. The classical controls use descriptor distance, the nearest/second-nearest distance ratio and mutual nearest-neighbor compatibility.

The ratio score measures how distinct the best descriptor neighbor is from the second best. Mutual matching requires the two descriptors to select one another. Neither condition proves geometric correctness, so a separate verification stage remains necessary.

Geometric verification uses robust estimation and COLMAP's verification tools to reject correspondences inconsistent with a two-view model.

There is a reproducibility distinction between implementations: the historical orchestration includes symmetric ratio scoring and custom verification, whereas the controlled reconstruction benchmarks use forward ratio with mutual top-1 and standard COLMAP verification. The numerical results must be reproduced with their actual benchmark scripts.

**Relevant files:** `src/underwater_vision/matching/matcher.py`, `src/underwater_vision/geometry/verification.py`, `scripts/match_features.py`.

## 7. Connect verified matches to COLMAP sparse reconstruction

The COLMAP adapter creates the database and writes calibrated cameras, images, keypoints and matches in the format expected by the mapper.

Sparse reconstruction then performs camera registration, point triangulation and bundle adjustment. Export utilities expose estimated camera-to-world poses and point-cloud statistics for subsequent evaluation.

The sparse-stage outputs include:

- Estimated camera poses.
- Registered-image count.
- Sparse RGB points and their observation tracks.
- Reprojection errors and track statistics.
- Exported point clouds and machine-readable metrics.

This stage makes reconstruction quality measurable. A visually plausible point cloud is useful for inspection, but it is not sufficient evidence of accurate geometry.

**Relevant files:** `src/underwater_vision/reconstruction/colmap/database.py`, `mapper.py`, `exporter.py`, `scripts/run_sfm.py`.

## 8. Add optional dense MVS and mesh inspection

Once sparse cameras are available, the optional dense stage uses a CUDA-enabled COLMAP executable for PatchMatch, fusion and meshing. This is distinct from the CPU SfM capabilities available through the local PyCOLMAP wheel.

Existing 16-image pilots produced dense clouds and meshes, including a classical mesh with approximately 260,000 faces. The mesh can be inspected independently of the training tasks.

Later support-based cleanup removes faces far from the observed fused cloud using a threshold scaled by local point spacing. On the existing pilots, approximately 68% of unsupported surface area was removed while the reported observed-point coverage was maintained.

That result measures support relative to the same observed cloud used to create the mesh. It does not demonstrate independent surface accuracy, and trimming can create holes or open boundaries. No new dense-MVS reconstruction was run during the latest learned-geometry experiments.

**Relevant files:** `scripts/setup_colmap.py`, `scripts/run_mvs.py`, `scripts/view_pointcloud.py`, `scripts/clean_mesh.py`, `src/underwater_vision/pointcloud/mesh_cleanup.py`.

## 9. Add deterministic visibility/reliability cues

Before introducing learned reliability, the system computes local image-quality evidence from contrast, sharpness, texture, entropy, exposure and color balance. These maps are sampled at keypoint coordinates.

Normalization uses fixed scales. Per-image min/max normalization is avoided because it could make a uniformly poor image appear to contain highly reliable regions.

The resulting reliability score is an engineering heuristic. It is not a direct physical measurement of underwater visibility or light attenuation. It is used as a candidate feature and ablation control, with its contribution evaluated experimentally.

An early learned-reliability experiment used geometric inlier frequency as supervision. The later v1 reference-supervised filters replace that label source with published-camera epipolar evidence to reduce circularity between labels and the pipeline's own RANSAC decisions.

**Relevant files:** `src/underwater_vision/visibility/deterministic.py`, `learned.py`, `consistency.py`, `scripts/estimate_visibility.py`, `scripts/train_reliability.py`.

## 10. Add frozen DINOv2 evidence and image-pair retrieval

DINOv2 is incorporated as a pretrained frozen representation. The project uses its evidence in two ways:

1. Local patch features are interpolated at SIFT coordinates, supplying appearance/context evidence alongside RootSIFT.
2. Global descriptors rank likely overlapping images, enabling pair retrieval beyond a fixed temporal window.

Retrieval normalizes descriptors, computes cosine similarities and selects top-k neighbors. It can be combined with sequential pairs. The motivation is to retain nonlocal connections when the trajectory revisits an area.

Retrieval evaluation uses a reference-camera co-frustum proxy. Poses alone do not provide observed overlap, occlusion or surface relief, so that proxy is not described as overlap ground truth.

DINOv2 is not presented as a complete feature matcher: SIFT still provides keypoint localization, and separate logic establishes correspondences.

**Relevant files:** `src/underwater_vision/features/dinov2.py`, `src/underwater_vision/retrieval/pairs.py`, `scripts/download_models.py`, `scripts/retrieve_pairs.py`.

## 11. Audit reference geometry before creating learned labels

Reference supervision is useful only if calibration and pose conventions are compatible with image coordinates. The audit therefore checks candidate matches against the fundamental matrix derived from the published cameras.

The procedure is:

1. Construct the intrinsic matrix at the prepared image resolution.
2. Apply the selected Brown distortion convention and undistort match endpoints.
3. Derive relative rotation and translation from the two reference camera-to-world poses.
4. Form the essential and fundamental matrices.
5. Measure Sampson epipolar error at the undistorted endpoints.

Conceptually:

```text
E = [t]_x R
F = inverse(K_b).T E inverse(K_a)
```

The selected convention yields a median audited error of **0.2934 px**, with **94.14% of audited matches below 1.5 px**.

This supports the chosen convention on the audited match population. Epipolar compatibility still does not prove that two endpoints represent the same physical point, and underwater refraction is not explicitly modeled.

**Relevant files:** `scripts/audit_reference.py`, `src/underwater_vision/geometry/reference.py`.

## 12. Establish spatial train/validation/test separation

The project moved beyond temporal-only separation because a moving camera can revisit the same physical region. Spatial partitions use reference camera positions to assign images to regions and enforce evaluation boundaries.

The roles are distinct:

- Training images fit model weights.
- Validation images choose checkpoints, thresholds or calibration parameters.
- Test images measure the selected method.

Evaluation guards reject image overlap with training/validation and incompatible checkpoint assumptions. Some experiments also enforce spatial region membership.

The separation is within one acquisition, not between independent scenes. Moreover, the latest reconstruction blocks have been examined repeatedly during development; their results are exploratory evidence rather than an untouched confirmatory test.

**Relevant files:** `src/underwater_vision/data/spatial.py`, `src/underwater_vision/research_pipeline.py`, `src/underwater_vision/matchability/protocol.py`.

## 13. Build v1 reference-supervised correspondence filters

The v1 classifier scores individual candidate correspondences using local descriptor evidence, ratio/mutual information, DINOv2 similarity, endpoint reliability and relative positions. Both a lightweight MLP and a context-attention network are implemented.

Reference geometry supplies positive and negative labels. The standard reference label thresholds are 1.5 px for positive and 4.0 px for negative, with an ambiguous interval excluded from ordinary binary classification.

Training and validation are separated spatially. Classification evaluation includes precision-recall, AP, ROC AUC and threshold-dependent metrics. The selected scores can subsequently filter matches before COLMAP.

On the held-out v1 candidate population:

| Scorer | AP |
|---|---:|
| Ratio-test scoring | 0.9233 |
| Reference-supervised MLP | 0.9473 |

The context model and official LightGlue adapter are separate comparisons. LightGlue is not the source of the MLP's AP figure, and the correct library name is LightGlue.

The original small reconstruction comparison did not establish a geometry gain: sparse points increased from 1,790 to 1,890, but reprojection error changed from approximately 0.4259 to 0.4504 px and ATE from 0.01043 to 0.01078 reference units. This motivated direct downstream evaluation rather than using AP as the reconstruction conclusion.

**Relevant files:** `scripts/train_match_filter.py`, `scripts/evaluate_match_filter.py`, `src/underwater_vision/matching/learned_filter.py`, `lightglue_adapter.py`, `src/underwater_vision/research_pipeline.py`.

## 14. Add learned confidence for reconstructed points

The confidence stage asks whether a point's observations contain evidence about its likely error. Candidate features include view count, track length, triangulation angle, reprojection error, reliability and cross-view appearance consistency.

Reference-camera triangulation and held-out-view reprojection provide error proxies. Held-out evaluation must avoid using the withheld observation to create the same prediction it evaluates.

Controls include the existing heuristic and a simpler view-count-only score. Evaluation uses:

- **Spearman correlation:** whether predicted and observed error rankings agree.
- **Sparsification curves:** whether removing low-confidence points reduces retained error.
- **AUSE:** the gap between predicted removal ordering and an oracle error ordering; lower is better.
- **Error calibration:** whether predicted error levels correspond to observed errors.

The learned point-confidence component is implemented, but has not consistently beaten the heuristic or established nominal uncertainty coverage. It remains a research component rather than a proven universal improvement.

**Relevant files:** `scripts/train_point_confidence.py`, `src/underwater_vision/pointcloud/learned_confidence.py`, `src/underwater_vision/evaluation/uncertainty.py`.

## 15. Build v3 dense matchability with SegFormer-B0

V3 changes the prediction unit from a candidate match to an image location. SegFormer-B0 takes RGB and predicts a one-channel matchability score map. The map can later be sampled at keypoints, used to rank correspondences or consumed independently of SfM.

Supervision remains sparse: reference-geometry correspondence evidence is attached to observed keypoints, rather than densely annotated semantic classes. Unknown or unsupported locations are not simply declared negative.

The dataset builder records the image graph, label protocol and preprocessing. Configuration includes sequential and retrieval connections. Training crops adjust keypoint coordinates, and photometric augmentation changes appearance without silently moving supervision locations.

SegFormer-B0 is compared with frozen DINOv2+MLP and other configured controls. The completed primary evaluation uses three spatial folds and seeds 42/43/44:

| Spatial fold | SegFormer mean AP | Frozen DINOv2+MLP mean AP |
|---|---:|---:|
| 0 | 0.885073 | 0.879971 |
| 1 | 0.840394 | 0.822597 |
| 2 | 0.886966 | 0.876570 |

Paired AP intervals support an advantage in folds 1 and 2, but not fold 0. The nine-run mean AP is **0.870811**. The range 0.84–0.89 describes fold means, not the minimum and maximum individual runs.

These are matchability classification results. They do not establish a corresponding camera-pose or mesh improvement.

**Relevant files:** `scripts/build_matchability_dataset.py`, `scripts/train_matchability.py`, `scripts/run_segformer_folds.py`, `scripts/evaluate_segformer_folds.py`, `src/underwater_vision/matchability/`, `configs/matchability/`.

## 16. Calibrate frozen matchability scores

After training, calibration addresses whether score magnitudes correspond to observed positive frequencies. It does not retrain the image network.

The controlled audit fits temperature scaling and positive-slope Platt scaling on the same 64 validation images per fold/seed. Platt scaling uses an affine logit transformation followed by sigmoid:

```text
p_calibrated = sigmoid(a * logit + b), with a > 0
```

The positive slope preserves score ordering, so AP remains unchanged. Calibration is stored as an optional checkpoint-hash-bound sidecar.

| Nine-run mean | Temperature | Platt |
|---|---:|---:|
| ECE, 10 bins | 10.60% | 3.50% |
| Brier score | 0.181301 | 0.165956 |
| Log-loss | 0.545001 | 0.507385 |

ECE improves in 8/9 runs; Brier and log-loss improve in 6/9. Seven of nine runs achieve ECE at most 5%. The calibration benefit is substantial on average, but not uniform across every fold and metric.

**Relevant files:** `scripts/benchmark_calibration.py`, `src/underwater_vision/matchability/calibration.py`, [controlled calibration audit](CONTROLLED_CALIBRATION_AUDIT_v1.md).

## 17. Add independent model inference and serving

The RGB matchability component was made usable outside reconstruction. PyTorch inference and ONNX inference produce score maps, and FastAPI exposes the model through an image interface.

Calibration is optional in both inference paths. Cache keys and provenance distinguish calibrated from uncalibrated predictions. Deployment assets include Docker configuration and API smoke scripts.

A real trained-model smoke check exercised ONNX/API prediction, a 720 × 960 uint16 map, keypoint sampling, health reporting and invalid-image rejection.

The demonstrated deployment scope is matchability inference. It should not be described as proof that the entire asynchronous SfM/MVS workflow is deployed through the API. TensorRT has not been demonstrated.

**Relevant files:** `scripts/export_onnx.py`, `scripts/serve_api.py`, `scripts/smoke_matchability_api.py`, `src/underwater_vision/serving/`, `Dockerfile.matchability`.

## 18. Measure reconstruction quality with controlled comparisons

Downstream evaluation uses three disjoint 32-image blocks: IDs 273–304, 599–630 and 1116–1147. Together they contain 96 distinct RGB images. The blocks satisfy the joint v1/SegFormer test-region requirements and cover two SegFormer folds, not all three.

Each block uses the same prepared images, fixed intrinsics, 171 window-6 pairs and mapper seeds 42/43/44. An initial comparison completed 54 reconstruction runs across full features and learned/heuristic selection variants.

The primary camera-center measurement is Sim3-aligned ATE. Because monocular reconstruction has unknown scale, estimated centers are aligned to the published reference centers before comparison. Units remain dataset reference units, not verified metres.

When methods register different cameras, ATE is compared on the same common camera population. Registration counts are reported separately. Paired uncertainty uses camera-cluster resampling and refits Sim3 in every draw.

Additional diagnostics include angular error, relative pose error, sparse point count, track length and reprojection error. Reprojection error and point count can use different point populations, so they are not substitutes for common-camera pose evaluation.

**Relevant files:** `scripts/benchmark_reconstruction.py`, `src/underwater_vision/evaluation/metrics.py`, `reconstruction_comparison.py`, `scripts/report_reconstruction_improvement.py`.

## 19. Improve the classical detector before claiming a DL gain

Lowering SIFT's contrast threshold from 0.04 to 0.01 admits weaker image structures that are useful in low-contrast underwater images. Nine detector runs test this change using the same benchmark blocks and seeds.

| Block | Registered cameras before → after | Sparse points before → after | Common-camera ATE reduction |
|---|---:|---:|---:|
| 273 | 32 → 32 | 5,644 → 11,502 | 20.24% |
| 599 | 22 → 32 | 1,145 → 12,214 | 27.07% |
| 1116 | 12 → 32 | 702 → 11,295 | 45.94% |

The first two paired ATE intervals exclude zero; the third does not. The common populations contain 32, 22 and 12 cameras respectively. These numbers must not be mixed with full 32-camera ATE values from the improved detector.

The requested 4,096-feature cap is unchanged, but actual extracted feature and match counts increase. Therefore this is not an equal-actual-feature-budget or speed improvement. Native reprojection errors also increase in two blocks as reconstruction coverage expands.

The setting is available as an optional profile. Existing filters trained on the original detector must not be silently treated as compatible with the new distribution.

**Relevant files:** `scripts/benchmark_detector_coverage.py`, `configs/experiment/low_contrast.yaml`, [reconstruction improvement report](RECONSTRUCTION_IMPROVEMENT_v1.md).

## 20. Rework learned selection to test its own geometric contribution

The latest development stage specifically addresses whether the complex learned components contribute beyond RootSIFT.

### 20.1 Preserve keypoints and equalize match budgets

Instead of deleting detector keypoints, the new conservative selection ranks only mutual ratio-0.8 candidate matches. Each pair keeps 90% of eligible matches, with a minimum of 128 when available. Pairs with fewer candidates keep all available matches.

Ratio ranking and random ranking retain exactly the same per-pair counts as the learned scorer. This separates useful learned ordering from the effect of simply removing 10% of matches.

### 20.2 Learn tighter reference precision

New MLP/context variants learn a soft target:

```text
target = exp(-0.5 * (reference_epipolar_error / 0.5px)^2)
```

Unlike ordinary binary labeling, this emphasizes finer epipolar precision. Checkpoints are selected by retained validation error. Original-pair and expanded-data models are evaluated separately, with expanded training using 226 training images and 65 validation images.

### 20.3 Learn on the stronger detector's actual survivor pool

A fresh conditional model is trained on contrast-0.01 RootSIFT candidates already surviving mutual ratio matching. Its ten inputs are descriptor distance, ratio, two local quality values, normalized endpoint coordinates and their displacement.

It uses **455,158 training candidates from 388 pairs**, with **221 validation pairs**. No DINOv2 or SegFormer score enters this model. Joint coordinate-axis relabeling provides augmentation; validation selects the context architecture at epoch 4 before test scoring.

These ten-input checkpoints are distinct from the older twelve-input v1 filters and use standalone controlled benchmark entry points.

### 20.4 Introduce triangulation observability into the target

A further model warm-starts the conditional context network and uses:

```text
utility = tight_epipolar_precision
          * min(1, (sin(reference_ray_angle) / sin(4 degrees))^2)
```

The reference ray angle is derived only from training/validation cameras. The 4-degree scale comes from the existing mapper initialization angle. Ten epochs are evaluated, and validation retained utility selects the checkpoint.

This target approximates triangulation information. It is not independently measured 3D point error, and its predicted scores are ranking utilities rather than calibrated correctness probabilities.

### 20.5 Retain all completed outcomes

The latest task completes **126 new SfM runs** across six experiment families, excluding the earlier 54 reconstruction comparisons and nine detector runs.

| Family | New runs |
|---|---:|
| Original-detector budget ranking | 54 |
| Expanded precision MLP | 18 |
| Expanded precision context | 9 |
| SegFormer with the strong detector | 27 |
| Conditional precision | 9 |
| Triangulation utility | 9 |

The strongest localized learned result is a reduction in median camera rotation error on block 599 from **1.6123° to 1.3228°**, or **17.95%**, while maintaining 32 registered cameras. The conditional model also beats ratio/random controls on this secondary metric, with unadjusted paired intervals excluding zero.

However, its ATE does not improve there, and the other two regions do not show the same angular advantage. On the old detector, frozen SegFormer reduces block-599 ATE by 2.36% against the full baseline, but that benefit does not survive comparison with the stronger detector baseline.

The observability model reduces mean ATE by approximately 3.28% and 2.76% in two blocks, but the intervals include zero; the remaining block becomes worse. Consequently, no learned method has yet established broad positional-accuracy improvement over the strongest classical controls.

**Relevant files:** `src/underwater_vision/matching/budget.py`, `geometry_precision.py`, `scripts/train_geometry_ranker.py`, `scripts/train_conditional_precision.py`, `scripts/train_observability_ranker.py`, `scripts/benchmark_geometry_ranking.py`, `scripts/benchmark_segformer_strong_baseline.py`, `scripts/benchmark_conditional_precision.py`.

## 21. Keep experiments modular and evidence traceable

The implementation preserves separate configuration switches for feature extraction, learned filtering, dense matchability, reliability and point confidence. New profiles do not automatically replace the default pipeline.

Experiment outputs record resolved settings, input images, model hashes, selection counts, status and metrics. The newer conditional/observability benchmarks archive their executed source. The original-budget benchmark was formatted after launch, so its archived formatted source is not the exact launch bytes; this limitation is recorded rather than concealed.

Reports and summaries are added under new versioned names. Older snapshots are useful evidence of what was known before each change.

Artifact cleanup previously removed verified redundant archives and obsolete caches, freeing approximately 7.13 GB, while checking retained current models. The latest learned-geometry task did not delete checkpoints or push commits.

At the latest source verification, **119 tests passed**, with Ruff and Black checks passing. Existing ONNX deprecation warnings remain. Tests support implementation correctness; the reconstruction experiments provide the empirical evidence for model usefulness.

## 22. Understand the full workflow and its present limits

The classical path can be read as:

```text
RGB images + calibration
  → prepare images
  → extract RootSIFT
  → select image pairs
  → descriptor matching
  → geometric verification
  → COLMAP sparse reconstruction
  → optional dense MVS and mesh
  → confidence, visualization and evaluation
```

Learned evidence enters through configurable branches: DINOv2 descriptors/retrieval, correspondence scoring, RGB matchability and point confidence. They have distinct supervision and output conventions; they should not be treated as a single universally validated configuration.

The current evidence supports:

- A functioning modular image-to-3D research pipeline.
- Better held-out match ranking with the v1 MLP.
- SegFormer matchability advantages in two of three spatial folds.
- Better average probability calibration.
- Stronger camera coverage and common-camera pose agreement from lower-contrast RootSIFT.
- A localized learned-filter improvement in camera orientation.

The remaining gaps are a complete 1,244-image SfM/MVS run, an untouched confirmatory region or acquisition, consistent learned positional-accuracy gains, and independently measured dense-surface accuracy. The next credible geometry claim requires locking the primary metric and model before evaluating new evidence, rather than continuing to select models from these same three development blocks.

## 23. Read the supporting reports

| Evidence | Report |
|---|---|
| Consolidated baseline/v1/v3 and earlier improvements | [PROJECT_SUMMARY_v4.md](PROJECT_SUMMARY_v4.md) |
| Original reference-supervised research | [docs/research_report_v1.md](docs/research_report_v1.md) |
| Completed SegFormer spatial folds | [docs/segformer_folds_report_final_v3.md](docs/segformer_folds_report_final_v3.md) |
| Controlled probability calibration | [CONTROLLED_CALIBRATION_AUDIT_v1.md](CONTROLLED_CALIBRATION_AUDIT_v1.md) |
| Strong classical reconstruction change | [RECONSTRUCTION_IMPROVEMENT_v1.md](RECONSTRUCTION_IMPROVEMENT_v1.md) |
| Latest complete learned-geometry assessment | [LEARNED_3D_CONTRIBUTION_v1.md](LEARNED_3D_CONTRIBUTION_v1.md) |
| SegFormer against the strong baseline | [SEGFORMER_STRONG_BASELINE_v1.md](SEGFORMER_STRONG_BASELINE_v1.md) |
| Conditional precision geometry | [CONDITIONAL_PRECISION_GEOMETRY_v1.md](CONDITIONAL_PRECISION_GEOMETRY_v1.md) |
| Observability-target geometry | [OBSERVABILITY_GEOMETRY_v1.md](OBSERVABILITY_GEOMETRY_v1.md) |

This walkthrough creates no new model, reconstruction result or deployment claim. It documents the implemented project and the evidence available at this snapshot.
