# Learned geometry assessment v1

This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. Detector features are identical between controls and learned scorers. All methods use contrast0.01, including the strong full baseline.

## Method and separation

One architecture/checkpoint selected solely using spatial validation before test scoring. Same contrast0.01 keypoints and immutable RAW mutual ratio0.8 match pools read from classical databases, not RANSAC inliers. Exact retained count per pair matches existing ratio90/random90 controls, minimum128 and90% retention. No test labels or poses select matches. All three locked32-image blocks and mapper seeds42/43/44; fixed intrinsics/COLMAP. Fresh conditional model learned precision on the same detector distribution, with joint coordinate relabeling. Already examined development regions, not a blind new scene.

The conditional precision model is trained from scratch on455,158 eligible matches from388 train pairs across226 images. Both MLP and context-attention are evaluated on221 validation pairs across65 images. Validation selects context epoch4 before test scoring. Inputs are descriptor distance/ratio, two local reliability values and normalized endpoint/relative coordinates; no DINO or SegFormer score enters this model. It learns exp(-0.5*(reference epipolar error/0.5px)^2) only within the matcher survivor pool. Joint coordinate-axis relabeling augments pair geometry. No test label fits weights or selects an architecture.

The conditional scorer uses the immutable classical RAW match pool, before COLMAP's geometric verification. It does not consume stored RANSAC inliers or reference test geometry. Recomputed descriptor neighbors are checked against frozen pairs.

Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline. Endpoint SegFormer evidence ranks correspondences rather than deleting keypoints.

## Native reconstruction statistics

Three-seed means. These point/reprojection metrics use each run's native point population.

| Block | Method | Registered | Sparse points | Reprojection px |
|---|---|---:|---:|---:|
| block273 | precision90 | 32.0 | 10685 | 0.4681 |
| block273 | ratio_full | 32.0 | 11502 | 0.4841 |
| block273 | ratio90 | 32.0 | 10479 | 0.4752 |
| block273 | random90 | 32.0 | 10903 | 0.4805 |
| block599 | precision90 | 32.0 | 11340 | 0.7336 |
| block599 | ratio_full | 32.0 | 12214 | 0.7671 |
| block599 | ratio90 | 32.0 | 11496 | 0.7536 |
| block599 | random90 | 32.0 | 11880 | 0.7630 |
| block1116 | precision90 | 32.0 | 10449 | 0.4785 |
| block1116 | ratio_full | 32.0 | 11295 | 0.5030 |
| block1116 | ratio90 | 32.0 | 10429 | 0.4933 |
| block1116 | random90 | 32.0 | 10965 | 0.5008 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | precision90 | ratio_full | 32 | 0.021124 → 0.020712 | 1.95% | [-0.000573, 0.000180] | True |
| block273 | precision90 | ratio90 | 32 | 0.019383 → 0.020712 | -6.86% | [-0.000010, 0.001574] | True |
| block273 | precision90 | random90 | 32 | 0.019991 → 0.020712 | -3.60% | [0.000076, 0.000873] | True |
| block599 | precision90 | ratio_full | 32 | 0.022459 → 0.022764 | -1.36% | [-0.000859, 0.001255] | True |
| block599 | precision90 | ratio90 | 32 | 0.022842 → 0.022764 | 0.34% | [-0.000936, 0.000753] | True |
| block599 | precision90 | random90 | 32 | 0.022345 → 0.022764 | -1.88% | [-0.000707, 0.001241] | True |
| block1116 | precision90 | ratio_full | 32 | 0.017402 → 0.017643 | -1.38% | [-0.000726, 0.001083] | True |
| block1116 | precision90 | ratio90 | 32 | 0.017168 → 0.017643 | -2.76% | [-0.000331, 0.001111] | True |
| block1116 | precision90 | random90 | 32 | 0.017462 → 0.017643 | -1.03% | [-0.000754, 0.001079] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

Existing v1 filters and SegFormer remain independent optional components. The geometry MLP uses the same 12-input feature convention and SIFT contrast0.04; its output is a ranking score, not a calibrated correctness probability. `experiment=geometry_ranked` activates the optional conservative match-budget path. `experiment=segformer_geometry` uses endpoint evidence. That historical orchestration samples calibrated probability maps, whereas the controlled scripts sample raw logits before sigmoid; use the scripts for exact numerical reproduction. The ordinary threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/train_geometry_ranker.py --output outputs/geometry_ranker_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_geometry_ranking.py --output outputs/geometry_ranking_reproduction_v1
```

Both outputs must be new. `benchmark_geometry_ranking.py --ranker <checkpoint>` selects the independently stored ranker. `--expanded` on the training script uses all 226 v1 training images and 65 validation images. `benchmark_segformer_strong_baseline.py` performs the contrast0.01 comparison without either learned MLP. All reports, models and run directories from this assessment are new; no commit or push is performed.
