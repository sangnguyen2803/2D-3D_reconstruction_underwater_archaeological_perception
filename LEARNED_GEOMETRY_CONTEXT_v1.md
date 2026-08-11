# Learned geometry assessment v1

This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. Detector features are identical between controls and learned scorers. All methods use contrast0.04.

## Method and separation

Frozen SIFT contrast0.04 features, same 171 window6 pairs and fixed intrinsics. Every scorer retains ceil(90% mutual ratio0.8 survivors), minimum128 or all available. Same retained count for every pair. SegFormer uses endpoint scores rather than dropping keypoints. No test pose or reference label selects matches. Geometry MLP checkpoint chosen using original validation pairs only. This is a fixed development comparison on already examined spatial test blocks.

A match network is initialized from its original v1 checkpoint and fine-tuned with a soft target exp(-0.5*(reference epipolar error/0.5px)^2). Twenty epochs are evaluated on spatial validation pairs only; the checkpoint minimizes clipped reference error of retained validation matches. This is tight epipolar compatibility supervision, not independent 3D ground truth. No SegFormer weights are retrained.

Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline. Endpoint SegFormer evidence ranks correspondences rather than deleting keypoints.

## Native reconstruction statistics

Three-seed means. These point/reprojection metrics use each run's native point population.

| Block | Method | Registered | Sparse points | Reprojection px |
|---|---|---:|---:|---:|
| block273 | tight90 | 32.0 | 5245 | 0.4876 |
| block273 | ratio_full | 32.0 | 5644 | 0.5057 |
| block273 | ratio90 | 32.0 | 5184 | 0.5006 |
| block273 | random90 | 32.0 | 5356 | 0.5004 |
| block599 | tight90 | 22.0 | 1104 | 0.6167 |
| block599 | ratio_full | 22.0 | 1145 | 0.6282 |
| block599 | ratio90 | 22.0 | 1090 | 0.6170 |
| block599 | random90 | 22.0 | 1126 | 0.6264 |
| block1116 | tight90 | 12.0 | 658 | 0.2973 |
| block1116 | ratio_full | 12.0 | 702 | 0.3120 |
| block1116 | ratio90 | 12.0 | 655 | 0.3050 |
| block1116 | random90 | 12.0 | 686 | 0.3108 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | tight90 | ratio_full | 32 | 0.026483 → 0.025832 | 2.46% | [-0.000920, -0.000045] | True |
| block273 | tight90 | ratio90 | 32 | 0.024106 → 0.025832 | -7.16% | [0.000222, 0.002248] | True |
| block273 | tight90 | random90 | 32 | 0.026459 → 0.025832 | 2.37% | [-0.001039, 0.000272] | True |
| block599 | tight90 | ratio_full | 22 | 0.015969 → 0.016163 | -1.21% | [-0.000508, 0.000611] | True |
| block599 | tight90 | ratio90 | 22 | 0.016516 → 0.016163 | 2.14% | [-0.000954, 0.000207] | True |
| block599 | tight90 | random90 | 22 | 0.016307 → 0.016163 | 0.89% | [-0.000648, 0.000274] | True |
| block1116 | tight90 | ratio_full | 12 | 0.007410 → 0.007823 | -5.58% | [-0.001450, 0.000512] | True |
| block1116 | tight90 | ratio90 | 12 | 0.007846 → 0.007823 | 0.29% | [-0.001471, 0.000221] | True |
| block1116 | tight90 | random90 | 12 | 0.007638 → 0.007823 | -2.43% | [-0.001454, 0.000501] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

Existing v1 filters and SegFormer remain independent optional components. The geometry MLP uses the same 12-input feature convention and SIFT contrast0.04; its output is a ranking score, not a calibrated correctness probability. `experiment=geometry_ranked` activates the optional conservative match-budget path. `experiment=segformer_geometry` uses endpoint evidence. That historical orchestration samples calibrated probability maps, whereas the controlled scripts sample raw logits before sigmoid; use the scripts for exact numerical reproduction. The ordinary threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/train_geometry_ranker.py --output outputs/geometry_ranker_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_geometry_ranking.py --output outputs/geometry_ranking_reproduction_v1
```

Both outputs must be new. `benchmark_geometry_ranking.py --ranker <checkpoint>` selects the independently stored ranker. `--expanded` on the training script uses all 226 v1 training images and 65 validation images. `benchmark_segformer_strong_baseline.py` performs the contrast0.01 comparison without either learned MLP. All reports, models and run directories from this assessment are new; no commit or push is performed.
