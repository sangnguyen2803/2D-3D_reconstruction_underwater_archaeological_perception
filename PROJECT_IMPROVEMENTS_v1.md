# Project improvements v1

This report records measured changes after the eight v1 publication commits. Existing summaries, benchmark artifacts, checkpoints and original meshes were preserved. New code has not been committed or pushed.

## Validation-only SegFormer confidence calibration

Added a standalone positive-slope affine sigmoid calibrator (Platt-style scaling). It fits unweighted binary NLL on validation logits; its slope is positive so it preserves score ordering. A JSON sidecar binds the parameters to the exact checkpoint SHA-256. The neural weights are unchanged.

This experiment uses 64 evenly spaced images from each locked validation slab, selected by a fixed compute budget. Each model is evaluated on its complete original test slab. All three spatial folds and seeds 42/43/44 are included. The temperature baseline was fitted on its full validation slab, so the calibration data budgets differ. The cap was introduced because the GPU ran at 210 MHz, not to optimize test scores. Two earlier full-validation exploratory runs are preserved separately; their results are not mixed into this nine-run aggregate.

| Metric (equal-weight mean over nine runs) | Saved temperature baseline | Validation-only Platt |
|---|---:|---:|
| ECE, 10 equal-width bins | 10.88% | 3.50% |
| Brier score | 0.181505 | 0.165956 |
| Average precision | 0.870811 | 0.870811 |

Relative change in mean ECE: **67.83% reduction**. ECE is a probability calibration error, not classification accuracy or matching AP.

| Fold | Seed | ECE before | ECE after | Brier before | Brier after |
|---|---:|---:|---:|---:|---:|
| 0 | 42 | 4.79% | 5.40% | 0.162468 | 0.164500 |
| 0 | 43 | 5.75% | 5.15% | 0.166926 | 0.169523 |
| 0 | 44 | 5.29% | 4.60% | 0.158941 | 0.161300 |
| 1 | 42 | 12.56% | 4.26% | 0.204475 | 0.181216 |
| 1 | 43 | 14.82% | 3.58% | 0.205047 | 0.177692 |
| 1 | 44 | 6.38% | 4.30% | 0.184413 | 0.178681 |
| 2 | 42 | 17.93% | 1.65% | 0.191319 | 0.154382 |
| 2 | 43 | 15.06% | 0.84% | 0.179230 | 0.152715 |
| 2 | 44 | 15.32% | 1.72% | 0.180722 | 0.153593 |

Runs meeting the original 5% ECE target: 1/9 before, 7/9 after.

Paired 95% confidence intervals for after minus before:

- ece: [-0.091672, -0.055275].
- brier: [-0.019561, -0.011558].

Equal-weight macro mean across nine runs. 500 paired image-cluster bootstrap draws, identical draws across seeds of each fold; fixed-fold conditional CIs. One acquisition with spatial slabs, not independent scenes. Calibration labels measure reference epipolar compatibility, not dense pixel truth.

The optional calibration is available to PyTorch and ONNX predictors, feature-map caching and the standalone FastAPI CLI via `--calibration`. Cache keys include the calibration hash. The default remains the original temperature path. A real trained ONNX model passed the in-process FastAPI smoke check: a 720x960 uint16 score map, 32 keypoints, health and invalid-image rejection. This is not a new Docker build or a latency benchmark.

Method reference: [Guo et al., On Calibration of Modern Neural Networks (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html). This experiment adds an intercept beyond the one-parameter temperature model.

Evidence: [nine-run metrics](outputs/calibration_subset64_v1/comparison_v1.json), [calibration figure](outputs/calibration_subset64_v1/calibration_comparison_v1.png), [ONNX/API smoke](outputs/calibration_subset64_v1/onnx_api_smoke_v1.json).

## Density-adaptive mesh support cleanup

Added optional postprocessing that estimates local point spacing using eight neighboring dense points. Faces are retained only when all three vertices lie within three local spacings of observed points. The original mesh/cloud are hash checked and left intact; the cleaned PLY preserves vertex colors. This removes extrapolated Poisson surface regions instead of filling unobserved regions.

| Pilot reconstruction | Faces before / after | Removed mesh area | Dense-point coverage before / after | P95 mesh-vertex distance to dense cloud before / after |
|---|---:|---:|---:|---:|
| classical | 260,523 / 240,482 | 68.11% | 86.23% / 86.23% | 0.13431 / 0.05768 |
| visibility | 267,663 / 247,295 | 68.03% | 87.00% / 87.00% | 0.14219 / 0.06259 |

**Coverage definition:** fraction of observed dense points within two local spacings of a retained mesh vertex. Coverage is unchanged relative to the original mesh; this does not mean 100% scene coverage. Removed area is surface area, not triangle count. Distances use the reconstruction coordinate scale, not independently verified meters.

Both meshes come from the existing 16-image pilot. The tests measure support against the fused cloud used for meshing, not independent ground-truth surface accuracy. This is a postprocessing improvement, not a proven SegFormer benefit to dense reconstruction. Trimmed meshes can have holes and open boundaries; they are not watertight assets.

![Original and trimmed mesh at the same view and scale](outputs/mesh_cleanup_visibility_v1/comparison_v1.png)

Evidence: [visibility mesh metrics](outputs/mesh_cleanup_visibility_v1/comparison_v1.json), [classical mesh metrics](outputs/mesh_cleanup_classical_v1/comparison_v1.json).

## Storage cleanup and validation

Removed 714 inspected files and freed **7,132,895,533 bytes / 7.13 GB**. Two redundant archives were removed after CRC/size verification of all 1,244 extracted photographs and 294 installed COLMAP files. Removed 712 obsolete DINO evaluation-cache NPZ files. All **64 current model/checkpoint artifacts** were preserved and SHA-256 verified. No current model was sufficiently redundant to justify deleting it.

Details: [cleanup report](docs/artifact_cleanup_v1.md), [verification](outputs/artifact_cleanup_v1/verification_v1.json).

Validation: **110 tests passed**; Ruff and Black passed on all source, scripts and tests. The paired-bootstrap implementation was checked against explicit image resampling with ignored labels. Existing ONNX tests emit deprecation warnings. The local ONNX optional dependency was moved into the correct TOML optional-dependency table; v1 GitHub history was not changed.

## Reproduction

```powershell
.venv\Scripts\python.exe scripts/benchmark_calibration.py --validation-limit 64 --output outputs/calibration_reproduction_v1
.venv\Scripts\python.exe scripts/clean_mesh.py --mesh outputs/visibility_gpu_eval/dense/mesh.ply --cloud outputs/visibility_gpu_eval/dense/fused.ply --output outputs/mesh_cleanup_reproduction_v1
```

Every output directory must be new. To apply a fitted calibrator to the service, use `scripts/serve_api.py --task matchability --checkpoint <matching checkpoint> --calibration <calibration_v1.json>`. ONNX mode accepts the same sidecar only when its source checkpoint hash matches deployment metadata.

## Concise CV wording

```latex
\item Improved \textbf{SegFormer-B0 confidence calibration}, reducing mean ECE from \textbf{10.88\% to 3.50\%} across \textbf{three spatial folds and three seeds} using validation-only Platt scaling.
\item Implemented \textbf{density-adaptive mesh cleanup}, removing \textbf{68\% of unsupported surface area} while preserving observed-point coverage on a \textbf{16-image reconstruction pilot}.
```

Choose the calibration bullet only as an aggregate calibration claim. It does not mean every run improved, every run passed the 5% target, AP increased, or full-scene SfM/MVS was completed. The mesh bullet describes the implemented support criterion, not externally validated ground-truth accuracy.
