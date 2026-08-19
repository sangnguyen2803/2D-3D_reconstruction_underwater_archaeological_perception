# Reconstruction improvement v1

Completed a 54-run learned-feature SfM ablation and a nine-run low-contrast detector ablation. The original 16-view artifacts and all historical reports remain intact. No new dense MVS or independent surface-accuracy claim is made.

The supported improvement is lowering the RootSIFT detector contrast threshold from 0.04 to 0.01: all three regions now register 32/32 cameras. Common-camera reference-aligned pose error decreases by 20.24% and 27.07% in the first two regions, with paired 95% intervals excluding zero. The third region gains 20 registered cameras, but its pose-error reduction has an interval crossing zero. This is a classical detector improvement; it is not attributed to SegFormer or learned matching.

## Controlled setup

Three filename-contiguous 32-image blocks in intersection of v1 and corresponding SegFormer test regions. Same 171 window-6 pairs and fixed intrinsics. 75% per-image keypoint budgets are identical for response/random/SegFormer selectors. SegFormer training seed and mapper seed paired across 42/43/44; v1 context weights fixed seed42. No reference poses or epipolar labels used to select matches or fit geometry.

First 32 images in the first qualifying contiguous intersection run for fold1, and first 32 images from each of the two qualifying runs for fold2. No qualifying 32-contiguous-image intersection exists for fold0. These regions were already examined in prior learning benchmarks.

Same immutable RGB960 pixels, 32-view blocks, window-6 pair graph, 4096 feature cap, mutual ratio0.8 matching, fixed calibration and COLMAP settings. Only SIFT contrast threshold changes. Actual detected feature/match counts and costs are explicitly reported. No reference pose or epipolar label fits or selects matches. This detector setting is a development hypothesis, not a blind new-scene validation.

The detector change is now optional in the source and `experiment=low_contrast` profile. The default remains SIFT contrast 0.04. Nondefault contrast enters the feature-cache key. Existing v1 match filters reject a changed contrast because their training detector distribution was 0.04. The historical orchestration also includes symmetric ratio scoring and custom verification; the controlled scripts here use forward ratio plus mutual top-1 and standard COLMAP verification. Use the supplied benchmark scripts for exact reproduction of the reported metrics.

## Detector change: native reconstruction statistics

Three-seed means; compare ATE on common camera sets in the next table, not different registration populations.

| Region | Registered before / after | Points before / after | Reprojection before / after (px) |
|---|---:|---:|---:|
| block273 | 32.0 / 32.0 | 5644 / 11502 | 0.5057 / 0.4841 |
| block599 | 22.0 / 32.0 | 1145 / 12214 | 0.6282 / 0.7671 |
| block1116 | 12.0 / 32.0 | 702 / 11295 | 0.3120 / 0.5030 |

## Detector change: identical-camera pose agreement

| Region | Common cameras | ATE before / after | Relative reduction | Paired 95% CI, after minus before |
|---|---:|---:|---:|---:|
| block273 | 32 | 0.026483 / 0.021124 | 20.24% | [-0.006397, -0.000782] |
| block599 | 22 | 0.015969 / 0.011647 | 27.07% | [-0.006071, -0.000950] |
| block1116 | 12 | 0.007410 / 0.004006 | 45.94% | [-0.004854, 0.000352] |

500 paired four-consecutive-camera cluster bootstrap draws; re-fit Sim3 in every draw. The same common camera set and draws are used across all three seeds. Missing-camera counts are reported separately. Fixed-region, already reconstructed-network conditional intervals, not independent scene uncertainty.

ATE uses published bundle-adjusted poses after similarity alignment; units are the dataset reference scale, not independently verified meters. This measures reference-pose agreement, not ground-truth surface accuracy. Native reprojection errors and point counts have different point populations and are not sufficient accuracy evidence on their own. Increased feature coverage costs additional extraction, descriptor matching and mapping compute; this is not a speed improvement.

## SegFormer selection: matched 75% per-image budgets

| Region | Control | Common-camera ATE reduction | Paired 95% CI, after minus before |
|---|---|---:|---:|
| block273 | ratio_full | 1.88% | [-0.001412, 0.000669] |
| block273 | response75_ratio | 8.54% | [-0.002947, -0.000437] |
| block273 | random75_ratio | 0.78% | [-0.001278, 0.001082] |
| block599 | ratio_full | 8.27% | [-0.002293, -0.000420] |
| block599 | response75_ratio | 14.65% | [-0.004190, -0.000615] |
| block599 | random75_ratio | 13.12% | [-0.003427, -0.000751] |
| block1116 | ratio_full | -73.18% | [0.000162, 0.006893] |
| block1116 | response75_ratio | -49.27% | [0.000220, 0.005013] |
| block1116 | random75_ratio | -23.36% | [-0.000330, 0.001894] |

All 54 runs are retained, including failures or regressions. The equal-budget SegFormer result must be assessed against both response and random selection, not only the full-feature baseline. Registration loss is an adverse outcome even when conditional common-camera ATE is lower. No test pose or reference label was used to choose a keypoint or a match. These are development experiments on one acquisition, not an untouched external benchmark.

An exploratory fixed-intrinsics Soft-L1 global BA probe on block273 lowered reprojection error but increased ATE from 0.026967 to 0.030424; it was not promoted into the pipeline. Its artifacts remain in `outputs/robust_ba_probe_v1/`.

## Evidence and reproduction

- [54-run metrics](outputs/reconstruction_improvement_v1/comparison_v1.json)
- [Nine detector runs](outputs/detector_coverage_v1/comparison_v1.json)
- [Paired assessment](outputs/reconstruction_assessment_v1/assessment_v1.json)
- [Registered cameras](outputs/reconstruction_assessment_v1/registration_v1.png)
- [Camera trajectories](outputs/reconstruction_assessment_v1/trajectories_v1.png)

```powershell
.venv\Scripts\python.exe scripts/benchmark_reconstruction.py --output outputs/reconstruction_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_detector_coverage.py --contrast 0.01 --output outputs/detector_reproduction_v1
```

Both benchmark output directories must be new. The detector benchmark defaults to the preserved controlled baseline. New code and reports have not been committed or pushed.

## Validation and CV wording

The source validation completed with 112 passing tests, Ruff passing, and Black confirming 163 source/script/test files need no formatting changes. Tests cover detector default compatibility, low-contrast feature extraction, invalid threshold rejection, similarity alignment and paired common-camera comparison. Existing ONNX export deprecation warnings remain.

Suggested concise CV bullet:

> Optimized low-contrast RootSIFT extraction, reducing reference-aligned pose error by 20.2% and doubling sparse 3D points on a 32-view benchmark across three seeds.

An alternative emphasizing coverage across all three regions:

> Improved underwater SfM coverage to 32/32 registered views in three spatial regions, up from 32/32, 22/32 and 12/32, using low-contrast RootSIFT extraction.

These claims describe 96 distinct images from the 1,244-image acquisition. They do not imply a completed 1,244-image reconstruction. More features and matches increase compute; higher point counts alone do not establish surface accuracy. The SegFormer selector regresses in the third region and is retained as a documented ablation rather than promoted as a universally better reconstruction setting.
