# Visibility-Aware Underwater 2D–3D Reconstruction — Project Summary v4

**Snapshot:** 9 October 2026, Europe/Paris.  
**Scope:** baseline reconstruction, reference-supervised v1, dense-matchability v3, and the latest calibration, mesh-support, storage and sparse-reconstruction improvements. V2 is excluded from this summary.  
**Version meaning:** v4 is a new consolidated report, not a newly trained v4 neural model. Historical summaries and results remain preserved.

## 1. Project purpose and current outcome

The project reconstructs underwater archaeological scenes from overlapping RGB images and studies correspondence reliability under low contrast, weak texture, uneven illumination and degraded visibility. Its implemented components cover image loading, feature extraction, pair selection, learned match filtering, geometric verification, sparse SfM, optional dense MVS, point confidence, visualization and matchability inference.

The latest supported geometric improvement comes from lowering the RootSIFT detector contrast threshold from **0.04 to 0.01**. Three 32-image regions now register **32/32 cameras each**, versus **32/32, 22/32 and 12/32** previously. Common-camera reference-aligned pose error decreases by **20.24% and 27.07%** in the first two regions, with paired 95% intervals excluding zero. This improvement is separate from SegFormer learning.

The project also establishes a learned matching-ranking result, a spatially evaluated dense-matchability result and better probability calibration. These are distinct evaluation tasks and should not be presented as one overall accuracy score.

## 2. Data, inputs and outputs

- **Dataset:** EPITA/LRE Mermaid underwater archaeology acquisition, containing **1,244 RGB photographs and 1,244 published reference camera poses**.
- **Image resolution:** 3,840 × 2,880 source images; standard reported comparison inputs are 960 × 720.
- **Additional inputs:** camera calibration, model/configuration files, pretrained DINOv2 features, optional SegFormer and learned-filter checkpoints.
- **Outputs:** cached local/global features, candidate and verified correspondences, dense matchability score maps, camera poses, sparse RGB point clouds, optional dense clouds/meshes, confidence estimates, metrics and figures.

Reference geometry supports label construction and evaluation. The new reconstruction benchmarks estimate cameras from image correspondences without using reference poses to fit geometry or select matches. Reference poses are bundle-adjusted on the same acquisition, so agreement with them is not independent surveyed ground-truth accuracy. Underwater refraction is not explicitly modeled.

Dense matchability produces an image score map supervised at labeled keypoint locations. It is not semantic segmentation with dense class annotations. The full dataset supports learning and label generation; it has not all been reconstructed together in a single completed 1,244-image SfM/MVS run.

## 3. Implemented system

| Component | Implementation and role |
|---|---|
| Data and configuration | Download/integrity checks, calibrated image loading, spatial partitions, Hydra configuration |
| Local features | OpenCV SIFT locations and RootSIFT descriptors, optional low-contrast detector |
| Visual representation | Frozen DINOv2 local evidence and global retrieval descriptors |
| Correspondences | Ratio/mutual matching, reliability features, MLP and context-attention filters, LightGlue comparison |
| Geometry | RANSAC/geometric verification and COLMAP sparse reconstruction adapter |
| Dense matchability | SegFormer-B0, sparse reference-epipolar supervision, spatial controls and calibration |
| 3D confidence | Heuristic and learned point-error/confidence evaluation, held-out-view evidence, AUSE/ranking diagnostics |
| Dense reconstruction | COLMAP CUDA PatchMatch/fusion/meshing for existing pilot runs |
| Serving and exports | PyTorch and ONNX matchability predictors, FastAPI image interface, existing Docker deployment assets |
| Experiment tooling | NumPy/SciPy, PyTorch, visualization, MLflow integration and automated tests |

ONNX inference has been exercised with a real trained model. TensorRT deployment has not been demonstrated. Existing deployment checks do not establish that the entire SfM/MVS workflow is exposed through the matchability API.

## 4. Established baseline, v1 and v3 evidence

### Reference audit and learned correspondence filtering

The selected Brown calibration and camera-to-world convention give a median audited epipolar error of **0.2934 px**, with **94.14% of audited matches below 1.5 px**. Labels come from reference geometry rather than the project's RANSAC decisions.

On the held-out v1 candidate pool, the **MLP reaches AP 0.9473**, versus **0.9233** for ratio-test scoring. Context-attention is also implemented; its result is separate from the MLP figure. LightGlue is a comparison method and is not the source of AP 0.9473. Better ranking does not automatically prove better 3D geometry. Learned point confidence has not consistently beaten the heuristic or established nominal uncertainty coverage.

### SegFormer-B0 dense-matchability benchmark

The primary comparison completed **three spatial folds × three seeds**, with matched controls on the same labeled populations.

| Fold | SegFormer mean AP | Frozen DINOv2+MLP mean AP | Paired AP difference 95% CI |
|---|---:|---:|---:|
| 0 | 0.885073 | 0.879971 | [-0.004757, 0.013552] |
| 1 | 0.840394 | 0.822597 | [0.011027, 0.025558] |
| 2 | 0.886966 | 0.876570 | [0.006791, 0.014572] |

SegFormer outperforms the strongest control in **two of three folds** with positive paired intervals. Mean AP over nine runs is **0.870811**. The CV range **0.84–0.89** describes fold means; individual runs include AP 0.830307. These are spatial regions within one acquisition, not independent scenes.

## 5. Improvement: validation-only probability calibration

A positive-slope affine sigmoid calibrator adds an intercept to temperature-style scaling. It fits validation logits using unweighted binary log-loss, leaves network weights unchanged and preserves score ordering. Parameters are stored in a checkpoint-hash-bound JSON sidecar.

The controlled audit fits both temperature scaling and Platt scaling on the **same 64 validation images per fold/seed**, then evaluates all nine frozen models on their full original test populations.

| Nine-run macro mean | Temperature, matched budget | Platt, same budget |
|---|---:|---:|
| ECE, 10 bins | 10.60% | 3.50% |
| Brier score | 0.181301 | 0.165956 |
| Log-loss | 0.545001 | 0.507385 |

Mean ECE decreases by **67.00%**. The paired 95% ECE difference interval is **[-0.088309, -0.053362]**. ECE improves in 8/9 runs; Brier and log-loss improve in 6/9, with all three fold0 seeds becoming slightly worse on those proper scoring rules. AP stays unchanged because ranking is preserved. Seven of nine runs achieve ECE ≤5%; the original all-runs calibration target remains unmet.

The earlier comparison against temperature fitted on the full validation slab reported **10.88% → 3.50%**. The CV uses **10.60% → 3.50%** because the matched-budget audit is more controlled.

Optional calibration is wired into PyTorch/ONNX prediction, feature-map cache keys and the FastAPI CLI. A real ONNX/API smoke check exercised a 720 × 960 uint16 score map, sampled keypoints, health reporting and invalid-image rejection. No new Docker build or latency benchmark was performed for this improvement.

## 6. Improvement: low-contrast RootSIFT sparse reconstruction

### Experimental design

The completed reconstruction assessment contains **54 learned-selection runs plus nine detector runs**. It uses three disjoint 32-image blocks (**96 distinct images**), seeds 42/43/44, identical RGB960 inputs, **171 window-6 pairs per block**, fixed calibration and fixed COLMAP settings.

These blocks lie in the intersection of the v1 test region and the corresponding SegFormer test regions. They cover **two SegFormer folds**, not all three: no qualifying 32-image contiguous joint-test block exists for fold0. The regions had already been examined in learning benchmarks; this is development evidence, not a blind external benchmark.

Only detector contrast changes in the nine-run ablation: **0.04 → 0.01**. The requested feature cap remains 4,096, but actual feature and match counts increase. OpenCV may slightly exceed the requested cap at tied responses. The improvement therefore costs more computation and is not an equal-actual-feature-budget or speed result.

### Measured before/after results

Values below are three-seed means. Pose comparisons use identical common camera sets; registration counts are reported separately.

| Region | Registered cameras before → after | Sparse points before → after | Common-camera ATE before → after | Reduction |
|---|---:|---:|---:|---:|
| block273 | 32 → 32 | 5,644 → 11,502 | 0.026483 → 0.021124 | 20.24% |
| block599 | 22 → 32 | 1,145 → 12,214 | 0.015969 → 0.011647 | 27.07% |
| block1116 | 12 → 32 | 702 → 11,295 | 0.007410 → 0.004006 | 45.94% |

ATE uses dataset reference units after similarity alignment, not independently verified meters. Common-camera populations are respectively 32, 22 and 12. It would be misleading to compare a full 32-camera ATE directly with a baseline 12- or 22-camera ATE.

Paired 95% intervals for ATE after minus before:

- block273: **[-0.006397, -0.000782]**.
- block599: **[-0.006071, -0.000950]**.
- block1116: **[-0.004854, 0.000352]**, crossing zero; its pose improvement remains statistically inconclusive.

The procedure uses 500 paired four-consecutive-camera cluster bootstrap draws, refits Sim3 in each draw and shares draws across seeds. Intervals condition on these fixed regions and reconstructed networks, not independent scene variation.

Native mean reprojection errors change from **0.5057 → 0.4841 px**, **0.6282 → 0.7671 px**, and **0.3120 → 0.5030 px**. The latter increases accompany expanded reconstruction populations and remain adverse diagnostics. More points alone do not establish geometric accuracy.

![Registered camera coverage before and after](outputs/reconstruction_assessment_v1/registration_v1.png)

### Source integration and learned-selection findings

The new detector setting is optional through `configs/experiment/low_contrast.yaml`; default contrast stays 0.04. Nondefault settings enter feature-cache keys. Existing v1 filters reject the changed detector distribution rather than silently reuse incompatible training assumptions.

The historical orchestration includes symmetric ratio scoring and custom verification. The controlled benchmark uses forward ratio with mutual top-1 and standard COLMAP verification. Exact numerical reproduction must use the benchmark scripts rather than assume the optional orchestration profile has an identical matching protocol.

The 54-run ablation compares full features, the v1 context filter, and response/random/SegFormer selection at matched 75% per-image feature budgets. SegFormer improves pose agreement over response selection in the first two blocks but regresses in the third. Against random selection, improvement is supported only in block599. It is not promoted as a universally better reconstruction setting.

An exploratory Soft-L1 bundle-adjustment probe lowered reprojection error but increased ATE **0.026967 → 0.030424**. It was not enabled in the pipeline. No new dense MVS reconstruction was run for the low-contrast improvement.

## 7. Improvement: observed-support mesh cleanup

Optional density-adaptive trimming removes faces whose vertices lie more than three local point spacings from observed dense points. It uses the existing fused cloud, preserves colors and leaves original meshes intact.

| Existing 16-image pilot | Faces before → after | Surface area removed | Observed-point coverage before → after |
|---|---:|---:|---:|
| Classical | 260,523 → 240,482 | 68.11% | 86.23% → 86.23% |
| Visibility | 267,663 → 247,295 | 68.03% | 87.00% → 87.00% |

For the visibility mesh, P95 vertex-to-cloud distance decreases **0.14219 → 0.06259** in reconstruction units. Removed area is not the proportion of faces removed. Coverage means observed dense points within two local spacings of a retained vertex, not complete scene coverage. This is support-based cleanup against the cloud used to create the mesh, not independent surface-accuracy validation. Trimmed meshes can contain holes and open boundaries.

## 8. Storage, validation and publication state

- Removed **714 verified redundant files**, freeing **7,132,895,533 bytes (7.13 GB)**: two archives and 712 obsolete DINO evaluation-cache files.
- Verified archive extraction before removal: all **1,244 images** and **294 installed COLMAP files** passed size/CRC checks.
- Preserved and SHA-256 verified **64 current model/checkpoint artifacts**. Earlier checkpoint cleanup is recorded separately and is not counted again.
- Source validation: **112 tests passed**; Ruff passed; Black confirmed **163 source/script/test files** need no changes. Existing ONNX deprecation warnings remain.
- The earlier eight v1 commits were published through **8a991d5**. Latest improvement code and these new reports have not been committed or pushed.

## 9. Updated CV and evidence index

The CV keeps the original supervision, SegFormer and correspondence-filter claims, and adds a concise, separate RootSIFT reconstruction result. Markdown and ready-to-copy LaTeX are in [CV_PROJECT_v4.md](CV_PROJECT_v4.md).

Evidence supporting this summary:

- [v1 geometry and matching report](docs/research_report_v1.md).
- [Completed SegFormer folds](docs/segformer_folds_report_final_v3.md).
- [Calibration, mesh and cleanup improvements](PROJECT_IMPROVEMENTS_v1.md).
- [Matched-budget calibration audit](CONTROLLED_CALIBRATION_AUDIT_v1.md).
- [Detailed reconstruction assessment](RECONSTRUCTION_IMPROVEMENT_v1.md).
- [Machine-readable paired reconstruction metrics](outputs/reconstruction_assessment_v1/assessment_v1.json).
- [Camera trajectories](outputs/reconstruction_assessment_v1/trajectories_v1.png).
- [Artifact cleanup verification](docs/artifact_cleanup_v1.md).

Next evidence gaps are independent-scene validation, a complete 1,244-image reconstruction, validation of the low-contrast setting with downstream learned filters, and independently measured dense-surface accuracy. Current results support useful engineering improvements without establishing those broader outcomes.
