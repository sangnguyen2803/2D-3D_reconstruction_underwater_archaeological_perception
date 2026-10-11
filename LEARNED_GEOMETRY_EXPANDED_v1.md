# Learned geometry assessment v1

This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. The detector remains SIFT contrast0.04: low-contrast RootSIFT is not responsible for any differences in this experiment.

## Method and separation

Frozen SIFT contrast0.04 features, same 171 window6 pairs and fixed intrinsics. Every scorer retains ceil(90% mutual ratio0.8 survivors), minimum128 or all available. Same retained count for every pair. SegFormer uses endpoint scores rather than dropping keypoints. No test pose or reference label selects matches. Geometry MLP checkpoint chosen using original validation pairs only. This is a fixed development comparison on already examined spatial test blocks.

A new MLP is initialized from the original v1 MLP and fine-tuned on original training pairs with a soft target exp(-0.5*(reference epipolar error/0.5px)^2). Twenty epochs are evaluated on original validation pairs only; the checkpoint minimizes clipped reference error of retained validation matches. This is tight epipolar compatibility supervision, not independent 3D ground truth. No SegFormer weights are retrained. Endpoint SegFormer scores rank correspondences rather than delete keypoints.

Primary newly trained scorer: tight90. Original MLP, SegFormer-only and product-fusion rows are separate ablations. Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline.

## Native reconstruction statistics

Three-seed means. These point/reprojection metrics use each run's native point population.

| Block | Method | Registered | Sparse points | Reprojection px |
|---|---|---:|---:|---:|
| block273 | tight90 | 32.0 | 5248 | 0.4924 |
| block273 | tight_segformer90 | 32.0 | 5296 | 0.4930 |
| block273 | ratio_full | 32.0 | 5644 | 0.5057 |
| block273 | ratio90 | 32.0 | 5184 | 0.5006 |
| block273 | random90 | 32.0 | 5356 | 0.5004 |
| block599 | tight90 | 22.0 | 1094 | 0.6122 |
| block599 | tight_segformer90 | 22.0 | 1089 | 0.6161 |
| block599 | ratio_full | 22.0 | 1145 | 0.6282 |
| block599 | ratio90 | 22.0 | 1090 | 0.6170 |
| block599 | random90 | 22.0 | 1126 | 0.6264 |
| block1116 | tight90 | 12.0 | 663 | 0.2992 |
| block1116 | tight_segformer90 | 12.0 | 661 | 0.3103 |
| block1116 | ratio_full | 12.0 | 702 | 0.3120 |
| block1116 | ratio90 | 12.0 | 655 | 0.3050 |
| block1116 | random90 | 12.0 | 686 | 0.3108 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | tight90 | ratio_full | 32 | 0.026483 → 0.024777 | 6.44% | [-0.002021, -0.000270] | True |
| block273 | tight90 | ratio90 | 32 | 0.024106 → 0.024777 | -2.78% | [-0.000031, 0.000855] | True |
| block273 | tight90 | random90 | 32 | 0.026459 → 0.024777 | 6.36% | [-0.002119, 0.000022] | True |
| block273 | tight_segformer90 | ratio_full | 32 | 0.026483 → 0.026396 | 0.33% | [-0.000820, 0.000597] | True |
| block273 | tight_segformer90 | ratio90 | 32 | 0.024106 → 0.026396 | -9.50% | [-0.000033, 0.002849] | True |
| block273 | tight_segformer90 | random90 | 32 | 0.026459 → 0.026396 | 0.24% | [-0.001031, 0.001004] | True |
| block599 | tight90 | ratio_full | 22 | 0.015969 → 0.015645 | 2.03% | [-0.000909, 0.000115] | True |
| block599 | tight90 | ratio90 | 22 | 0.016516 → 0.015645 | 5.27% | [-0.001426, -0.000032] | True |
| block599 | tight90 | random90 | 22 | 0.016307 → 0.015645 | 4.06% | [-0.001099, -0.000021] | True |
| block599 | tight_segformer90 | ratio_full | 22 | 0.015969 → 0.015860 | 0.69% | [-0.000697, 0.000246] | True |
| block599 | tight_segformer90 | ratio90 | 22 | 0.016516 → 0.015860 | 3.97% | [-0.000899, -0.000256] | True |
| block599 | tight_segformer90 | random90 | 22 | 0.016307 → 0.015860 | 2.74% | [-0.000648, -0.000174] | True |
| block1116 | tight90 | ratio_full | 12 | 0.007410 → 0.007878 | -6.31% | [-0.001445, 0.000641] | True |
| block1116 | tight90 | ratio90 | 12 | 0.007846 → 0.007878 | -0.40% | [-0.001466, 0.000320] | True |
| block1116 | tight90 | random90 | 12 | 0.007638 → 0.007878 | -3.14% | [-0.001449, 0.000629] | True |
| block1116 | tight_segformer90 | ratio_full | 12 | 0.007410 → 0.008184 | -10.44% | [0.000082, 0.000774] | True |
| block1116 | tight_segformer90 | ratio90 | 12 | 0.007846 → 0.008184 | -4.30% | [-0.000079, 0.000737] | True |
| block1116 | tight_segformer90 | random90 | 12 | 0.007638 → 0.008184 | -7.15% | [0.000098, 0.000754] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

Existing v1 filters and SegFormer remain independent optional components. The new MLP uses the same 12-input feature convention, original image preprocessing and SIFT contrast0.04. Its score is a tight-geometry ranking score, not a calibrated correctness probability. `experiment=geometry_ranked` activates the optional conservative match-budget path; the ordinary threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/train_geometry_ranker.py --output outputs/geometry_ranker_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_geometry_ranking.py --output outputs/geometry_ranking_reproduction_v1
```

Both outputs must be new. The benchmark currently references the preserved ranker checkpoint path; reproduce it by configuring that path in the script rather than overwriting existing artifacts. All reports, models and run directories from this assessment are new; no commit or push is performed.
