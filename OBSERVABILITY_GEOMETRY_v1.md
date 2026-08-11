# Learned geometry assessment v1

This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. Detector features are identical between controls and learned scorers. All methods use contrast0.01, including the strong full baseline.

## Method and separation

One architecture/checkpoint selected solely using spatial validation before test scoring. Same contrast0.01 keypoints and immutable RAW mutual ratio0.8 match pools read from classical databases, not RANSAC inliers. Exact retained count per pair matches existing ratio90/random90 controls, minimum128 and90% retention. No test labels or poses select matches. All three locked32-image blocks and mapper seeds42/43/44; fixed intrinsics/COLMAP. Fresh conditional model learned precision on the same detector distribution, with joint coordinate relabeling. Already examined development regions, not a blind new scene.

Conditional geometry inputs contain descriptor distance/ratio, two local reliability values and normalized endpoint/relative coordinates; no DINO or SegFormer score enters this model. The source dataset has455,158 eligible matches from388 train pairs across226 images, and221 validation pairs across65 images. The initial precision model selects context epoch4 on validation. An observability checkpoint, if specified, warm-starts that architecture and maximizes validation retained utility after ten epochs. Reference-derived ray-angle utility is a proxy, not independent3D ground truth. Target used in this run: tight_epipolar_precision * min(1,(sin(reference_ray_angle)/sin(4deg))^2)

The conditional scorer uses the immutable classical RAW match pool, before COLMAP's geometric verification. It does not consume stored RANSAC inliers or reference test geometry. Recomputed descriptor neighbors are checked against frozen pairs.

Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline. Endpoint SegFormer evidence ranks correspondences rather than deleting keypoints.

## Native reconstruction statistics

Three-seed means. These point/reprojection metrics use each run's native point population.

| Block | Method | Registered | Sparse points | Reprojection px |
|---|---|---:|---:|---:|
| block273 | precision90 | 32.0 | 10662 | 0.4686 |
| block273 | ratio_full | 32.0 | 11502 | 0.4841 |
| block273 | ratio90 | 32.0 | 10479 | 0.4752 |
| block273 | random90 | 32.0 | 10903 | 0.4805 |
| block599 | precision90 | 32.0 | 11390 | 0.7370 |
| block599 | ratio_full | 32.0 | 12214 | 0.7671 |
| block599 | ratio90 | 32.0 | 11496 | 0.7536 |
| block599 | random90 | 32.0 | 11880 | 0.7630 |
| block1116 | precision90 | 32.0 | 10483 | 0.4780 |
| block1116 | ratio_full | 32.0 | 11295 | 0.5030 |
| block1116 | ratio90 | 32.0 | 10429 | 0.4933 |
| block1116 | random90 | 32.0 | 10965 | 0.5008 |

## Common-camera pose comparisons

ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.

| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |
|---|---|---|---:|---:|---:|---|---|
| block273 | precision90 | ratio_full | 32 | 0.021124 → 0.020432 | 3.28% | [-0.001047, 0.000318] | True |
| block273 | precision90 | ratio90 | 32 | 0.019383 → 0.020432 | -5.41% | [-0.000004, 0.001304] | True |
| block273 | precision90 | random90 | 32 | 0.019991 → 0.020432 | -2.20% | [-0.000016, 0.000895] | True |
| block599 | precision90 | ratio_full | 32 | 0.022459 → 0.022678 | -0.98% | [-0.000660, 0.000858] | True |
| block599 | precision90 | ratio90 | 32 | 0.022842 → 0.022678 | 0.72% | [-0.000655, 0.000452] | True |
| block599 | precision90 | random90 | 32 | 0.022345 → 0.022678 | -1.49% | [-0.000434, 0.000918] | True |
| block1116 | precision90 | ratio_full | 32 | 0.017402 → 0.016922 | 2.76% | [-0.001303, 0.001170] | True |
| block1116 | precision90 | ratio90 | 32 | 0.017168 → 0.016922 | 1.43% | [-0.000909, 0.001281] | True |
| block1116 | precision90 | random90 | 32 | 0.017462 → 0.016922 | 3.09% | [-0.001320, 0.001242] | True |

## Interpretation and reproducibility

A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.

The standalone benchmark is the exact numerical reproduction entry point. Conditional rankers have ten inputs and contrast0.01; older tight rankers have twelve inputs and contrast0.04. They are different checkpoints and must not be interchanged. All outputs are ranking scores, not calibrated probabilities. Existing v1 filters and SegFormer remain optional components. Historical endpoint orchestration samples calibrated probability maps; controlled SegFormer scripts sample raw logits before sigmoid. The default threshold path remains unchanged.

```powershell
.venv\Scripts\python.exe scripts/benchmark_conditional_precision.py --training <conditional-or-observability-training-folder> --output <new-run-folder>
```

Both outputs must be new. `benchmark_geometry_ranking.py --ranker <checkpoint>` selects the independently stored ranker. `--expanded` on the training script uses all 226 v1 training images and 65 validation images. `benchmark_segformer_strong_baseline.py` performs the contrast0.01 comparison without either learned MLP. All reports, models and run directories from this assessment are new; no commit or push is performed.
