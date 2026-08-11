# Learned geometry assessment v1

This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. Detector features are identical between controls and learned scorers. All methods use contrast0.01, including the strong full baseline.

## Method and separation

Compare SegFormer endpoint match ranking with the already improved contrast0.01 RootSIFT baseline, on identical cached keypoints/descriptors/RGB960 inputs, 171 window6 pairs and fixed intrinsics/COLMAP settings. Ratio90, random90 and SegFormer90 retain exactly the same match count in each pair: ceil(90% mutual ratio0.8 survivors), minimum128 or all available. Frozen corresponding held-out fold/seed SegFormer predicts raw logits at new detector locations. No reference pose/label selects matches. No new network training or tuning in this experiment. These are already examined development regions, not blind scenes.

SegFormer weights are frozen. CPU float32 full-resolution logits are sampled at the new contrast0.01 keypoint coordinates, then sigmoid endpoint scores rank matches. Models were originally supervised at contrast0.04 keypoints; no claim is made about classification accuracy under this detector shift. No network or score parameter is fitted on these test images.

Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline. Endpoint SegFormer evidence ranks correspondences rather than deleting keypoints.

## Native reconstruction statistics

Three-seed means. These point/reprojection metrics use each run's native point population.

| Block | Method | Registered | Sparse points | Reprojection px |
|---|---|---:|---:|---:|
| block273 | ratio90 | 32.0 | 10479 | 0.4752 |
| block273 | random90 | 32.0 | 10903 | 0.4805 |
| block273 | segformer90 | 32.0 | 10586 | 0.4681 |
| block273 | ratio_full | 32.0 | 11502 | 0.4841 |
| block599 | ratio90 | 32.0 | 11496 | 0.7536 |
| block599 | random90 | 32.0 | 11880 | 0.7630 |
| block599 | segformer90 | 32.0 | 11184 | 0.7511 |
| block599 | ratio_full | 32.0 | 12214 | 0.7671 |
| block1116 | ratio90 | 32.0 | 10429 | 0.4933 |
| block1116 | random90 | 32.0 | 10965 | 0.5008 |
| block1116 | segformer90 | 32.0 | 10259 | 0.4832 |
| block1116 | ratio_full | 32.0 | 11295 | 0.5030 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | segformer90 | ratio_full | 32 | 0.021124 → 0.020529 | 2.82% | [-0.000802, 0.000286] | True |
| block273 | segformer90 | ratio90 | 32 | 0.019383 → 0.020529 | -5.92% | [0.000140, 0.001332] | True |
| block273 | segformer90 | random90 | 32 | 0.019991 → 0.020529 | -2.69% | [0.000183, 0.000732] | True |
| block599 | segformer90 | ratio_full | 32 | 0.022459 → 0.023515 | -4.70% | [-0.000098, 0.001626] | True |
| block599 | segformer90 | ratio90 | 32 | 0.022842 → 0.023515 | -2.95% | [-0.000159, 0.001052] | True |
| block599 | segformer90 | random90 | 32 | 0.022345 → 0.023515 | -5.24% | [0.000022, 0.001646] | True |
| block1116 | segformer90 | ratio_full | 32 | 0.017402 → 0.020101 | -15.51% | [0.000499, 0.003265] | True |
| block1116 | segformer90 | ratio90 | 32 | 0.017168 → 0.020101 | -17.08% | [0.000424, 0.003469] | True |
| block1116 | segformer90 | random90 | 32 | 0.017462 → 0.020101 | -15.11% | [0.000555, 0.003161] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

Existing v1 filters and SegFormer remain independent optional components. The geometry MLP uses the same 12-input feature convention and SIFT contrast0.04; its output is a ranking score, not a calibrated correctness probability. `experiment=geometry_ranked` activates the optional conservative match-budget path. `experiment=segformer_geometry` uses endpoint evidence. That historical orchestration samples calibrated probability maps, whereas the controlled scripts sample raw logits before sigmoid; use the scripts for exact numerical reproduction. The ordinary threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/train_geometry_ranker.py --output outputs/geometry_ranker_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_geometry_ranking.py --output outputs/geometry_ranking_reproduction_v1
```

Both outputs must be new. `benchmark_geometry_ranking.py --ranker <checkpoint>` selects the independently stored ranker. `--expanded` on the training script uses all 226 v1 training images and 65 validation images. `benchmark_segformer_strong_baseline.py` performs the contrast0.01 comparison without either learned MLP. All reports, models and run directories from this assessment are new; no commit or push is performed.
