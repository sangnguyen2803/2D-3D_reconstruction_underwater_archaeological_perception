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
| block273 | ratio90 | 32.0 | 5184 | 0.5006 |
| block273 | random90 | 32.0 | 5356 | 0.5004 |
| block273 | mlp90 | 32.0 | 5241 | 0.4987 |
| block273 | tight90 | 32.0 | 5232 | 0.4963 |
| block273 | segformer90 | 32.0 | 5250 | 0.4900 |
| block273 | tight_segformer90 | 32.0 | 5283 | 0.4953 |
| block273 | ratio_full | 32.0 | 5644 | 0.5057 |
| block599 | ratio90 | 22.0 | 1090 | 0.6170 |
| block599 | random90 | 22.0 | 1126 | 0.6264 |
| block599 | mlp90 | 22.0 | 1098 | 0.6186 |
| block599 | tight90 | 22.0 | 1095 | 0.6182 |
| block599 | segformer90 | 22.0 | 1092 | 0.6292 |
| block599 | tight_segformer90 | 22.0 | 1092 | 0.6221 |
| block599 | ratio_full | 22.0 | 1145 | 0.6282 |
| block1116 | ratio90 | 12.0 | 655 | 0.3050 |
| block1116 | random90 | 12.0 | 686 | 0.3108 |
| block1116 | mlp90 | 12.0 | 651 | 0.2996 |
| block1116 | tight90 | 12.0 | 654 | 0.3025 |
| block1116 | segformer90 | 12.0 | 664 | 0.3128 |
| block1116 | tight_segformer90 | 12.0 | 661 | 0.3078 |
| block1116 | ratio_full | 12.0 | 702 | 0.3120 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | mlp90 | ratio_full | 32 | 0.026483 → 0.024807 | 6.33% | [-0.001979, -0.000270] | True |
| block273 | mlp90 | ratio90 | 32 | 0.024106 → 0.024807 | -2.91% | [-0.000083, 0.000914] | True |
| block273 | mlp90 | random90 | 32 | 0.026459 → 0.024807 | 6.24% | [-0.002046, -0.000025] | True |
| block273 | tight90 | ratio_full | 32 | 0.026483 → 0.024688 | 6.78% | [-0.002126, -0.000349] | True |
| block273 | tight90 | ratio90 | 32 | 0.024106 → 0.024688 | -2.42% | [-0.000209, 0.000763] | True |
| block273 | tight90 | random90 | 32 | 0.026459 → 0.024688 | 6.69% | [-0.002222, -0.000141] | True |
| block273 | segformer90 | ratio_full | 32 | 0.026483 → 0.026904 | -1.59% | [-0.000079, 0.000590] | True |
| block273 | segformer90 | ratio90 | 32 | 0.024106 → 0.026904 | -11.61% | [0.000243, 0.003394] | True |
| block273 | segformer90 | random90 | 32 | 0.026459 → 0.026904 | -1.68% | [-0.000182, 0.000802] | True |
| block273 | tight_segformer90 | ratio_full | 32 | 0.026483 → 0.026749 | -1.00% | [-0.000246, 0.000524] | True |
| block273 | tight_segformer90 | ratio90 | 32 | 0.024106 → 0.026749 | -10.96% | [0.000103, 0.003173] | True |
| block273 | tight_segformer90 | random90 | 32 | 0.026459 → 0.026749 | -1.09% | [-0.000399, 0.000842] | True |
| block599 | mlp90 | ratio_full | 22 | 0.015969 → 0.016209 | -1.50% | [-0.000153, 0.000505] | True |
| block599 | mlp90 | ratio90 | 22 | 0.016516 → 0.016209 | 1.86% | [-0.000431, 0.000055] | True |
| block599 | mlp90 | random90 | 22 | 0.016307 → 0.016209 | 0.60% | [-0.000231, 0.000147] | True |
| block599 | tight90 | ratio_full | 22 | 0.015969 → 0.016025 | -0.35% | [-0.000378, 0.000412] | True |
| block599 | tight90 | ratio90 | 22 | 0.016516 → 0.016025 | 2.97% | [-0.000655, -0.000095] | True |
| block599 | tight90 | random90 | 22 | 0.016307 → 0.016025 | 1.73% | [-0.000418, -0.000014] | True |
| block599 | segformer90 | ratio_full | 22 | 0.015969 → 0.015593 | 2.36% | [-0.000740, -0.000167] | True |
| block599 | segformer90 | ratio90 | 22 | 0.016516 → 0.015593 | 5.59% | [-0.001063, -0.000424] | True |
| block599 | segformer90 | random90 | 22 | 0.016307 → 0.015593 | 4.38% | [-0.000879, -0.000336] | True |
| block599 | tight_segformer90 | ratio_full | 22 | 0.015969 → 0.015736 | 1.46% | [-0.000561, -0.000023] | True |
| block599 | tight_segformer90 | ratio90 | 22 | 0.016516 → 0.015736 | 4.72% | [-0.000863, -0.000329] | True |
| block599 | tight_segformer90 | random90 | 22 | 0.016307 → 0.015736 | 3.50% | [-0.000684, -0.000283] | True |
| block1116 | mlp90 | ratio_full | 12 | 0.007410 → 0.007910 | -6.75% | [-0.001465, 0.000691] | True |
| block1116 | mlp90 | ratio90 | 12 | 0.007846 → 0.007910 | -0.81% | [-0.001486, 0.000295] | True |
| block1116 | mlp90 | random90 | 12 | 0.007638 → 0.007910 | -3.56% | [-0.001469, 0.000680] | True |
| block1116 | tight90 | ratio_full | 12 | 0.007410 → 0.007938 | -7.13% | [-0.001556, 0.000807] | True |
| block1116 | tight90 | ratio90 | 12 | 0.007846 → 0.007938 | -1.18% | [-0.001577, 0.000411] | True |
| block1116 | tight90 | random90 | 12 | 0.007638 → 0.007938 | -3.94% | [-0.001560, 0.000796] | True |
| block1116 | segformer90 | ratio_full | 12 | 0.007410 → 0.007765 | -4.79% | [0.000057, 0.000606] | True |
| block1116 | segformer90 | ratio90 | 12 | 0.007846 → 0.007765 | 1.03% | [-0.000120, 0.000585] | True |
| block1116 | segformer90 | random90 | 12 | 0.007638 → 0.007765 | -1.67% | [-0.000009, 0.000602] | True |
| block1116 | tight_segformer90 | ratio_full | 12 | 0.007410 → 0.007740 | -4.46% | [-0.000051, 0.000699] | True |
| block1116 | tight_segformer90 | ratio90 | 12 | 0.007846 → 0.007740 | 1.35% | [-0.000224, 0.000678] | True |
| block1116 | tight_segformer90 | random90 | 12 | 0.007638 → 0.007740 | -1.34% | [-0.000035, 0.000695] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

Existing v1 filters and SegFormer remain independent optional components. The new MLP uses the same 12-input feature convention, original image preprocessing and SIFT contrast0.04. Its score is a tight-geometry ranking score, not a calibrated correctness probability. `experiment=geometry_ranked` activates the optional conservative match-budget path; the ordinary threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/train_geometry_ranker.py --output outputs/geometry_ranker_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_geometry_ranking.py --output outputs/geometry_ranking_reproduction_v1
```

Both outputs must be new. The benchmark currently references the preserved ranker checkpoint path; reproduce it by configuring that path in the script rather than overwriting existing artifacts. All reports, models and run directories from this assessment are new; no commit or push is performed.
