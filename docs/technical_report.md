# Technical report

## Actual development results

All tabled runs use the same **16 images: acquisition IDs 0,2,...,30**, resolution 960 × 720,
calibration, pair window, ratio threshold, and seed 42. Neural B/C/D/S runs use the
CUDA PyTorch implementation. SIFT uses CPU OpenCV; SfM uses CPU PyCOLMAP 3.13.0;
dense reconstruction uses official COLMAP 4.2.1 CUDA on an RTX 4060 Laptop GPU (8 GB).

| Run | Cameras | Sparse points | Inlier ratio | Reprojection (px) | ATE (reference units) |
|---|---:|---:|---:|---:|---:|
| classical_eval | 16 | 1790 | 0.9509 | 0.4259 | 0.01043 |
| neural_gpu_eval | 16 | 1047 | 0.9531 | 0.3849 | 0.01146 |
| visibility_gpu_eval | 16 | 1065 | 0.9780 | 0.3933 | 0.01090 |
| consistency_gpu_eval | 16 | 1067 | 0.9778 | 0.3942 | 0.01099 |
| learned_eval | 16 | 1069 | 0.9770 | 0.3949 | 0.01210 |
| small_eval | 16 | 1359 | 0.9812 | 0.3997 | 0.01085 |
| semantic_gpu_eval | 0 | 0 | 0.3466 | Unavailable | Unavailable |


The pure-DINO descriptor ablation (`semantic_gpu_eval`) did not register a model.
The zero camera/point counts record this observed failure, not an invented reconstruction.
The earlier `semantic_only_eval` development run reused a stale matching-cache key and
must not be used for conclusions. The cache key now includes the semantic fusion weight;
the corrected pure-DINO run is the one in this table.

| Run | Dense points | Depth-supported points | Mesh vertices | Mesh faces |
|---|---:|---:|---:|---:|
| classical_eval | 220038 | 220036 | 130379 | 260523 |
| visibility_gpu_eval | 218756 | 218756 | 133945 | 267663 |


Interpretation: reliability raises self-verified inlier ratios, but does not dominate all
metrics. The classical baseline retains substantially more sparse points. Compared with
the hybrid neural baseline, handcrafted reliability yields slightly more points and
higher inlier ratio but slightly worse mean reprojection error. The learned head does
not establish a consistent advantage over handcrafted reliability. S/14 preserves more
points than B/14 on this block. These are measured development findings, not a SOTA claim.

## Learned reliability checkpoint

`outputs/models/reliability.pt` was trained for 20 epochs from acquisition IDs
200,202,...,262, separately from the evaluation block. It uses **1509 weakly
supervised observations**, 1073 training examples and
282 validation examples, with a discarded boundary group.
Best validation BCE is 0.4101. This is validation against
geometric soft labels, not independently calibrated reconstruction correctness.

## Where to look

| Artifact | Location |
|---|---|
| Setup and commands | `README.md` |
| Complete original dataset | `data/raw/mermaid/SR202204_LDM-S_D01/` |
| Dataset integrity records | `outputs/dataset_inventory.json`, `outputs/dataset_integrity.json` |
| Controlled comparison | `outputs/comparison/comparison.md` and `comparison.json` |
| Offline interactive demo | `outputs/demo.html` |
| Baseline sparse/dense/mesh | `outputs/classical_eval/` |
| Proposed sparse/dense/mesh | `outputs/visibility_gpu_eval/` |
| Neural, consistency, learned, small-backbone runs | `outputs/*_eval/` |
| Dense confidence | `dense/fused_confidence.ply` inside the baseline/proposed runs |
| Sparse confidence | `sparse_confidence.ply` and `.npz` inside each successful run |
| Checkpoint and training curve | `outputs/models/reliability.pt`, `.json` |
| MLflow runs | `mlruns/` |
| Method, protocol, limitations | `docs/` |
| Analysis notebooks | `notebooks/01_...` through `05_...` |

## Reproduce the comparison

```powershell
.\.venv\Scripts\python.exe scripts/run_ablation_suite.py --prefix reproduction --include-small --tracking
.\.venv\Scripts\python.exe scripts/setup_colmap.py
.\.venv\Scripts\python.exe scripts/run_mvs.py --run outputs/reproduction_classical --colmap "$PWD/tools/colmap/bin/colmap.exe"
.\.venv\Scripts\python.exe scripts/finalize_dense.py --run outputs/reproduction_classical
.\.venv\Scripts\python.exe -m pytest -q
```

Each new experiment name creates a new directory; existing experiment names are rejected.
See README.md for individual stages, visualization and custom image-block selections.

## Scope and remaining research work

- [ ] Full-scene reconstruction/benchmark across all 1,244 views. The data is complete;
  measured reconstruction results currently cover the development subset above.
- [ ] Multiple spatially verified trajectory blocks, repeated seeds, uncertainty estimates,
  and locked hyperparameter selection. Temporal blocks alone do not prove spatial separation.
- [ ] Independent camera calibration and reference-pose convention validation.
- [ ] Independent dense surface ground truth for accuracy/completeness/Chamfer/F-score.
- [ ] Confidence calibration against independently measured geometric errors.
- [ ] Optional object segmentation, Point Transformer, or FastAPI deployment. These were
  not added because the geometry core and truthful evaluation take priority.

The specification's full research definition of done is therefore **not yet a completed
full-scene research benchmark**. The implemented core, data download, runnable experiments,
real sparse/dense reconstructions, confidence maps, and measured development report exist.

## Failures and corrections

An interrupted HTTP sample extraction left one incomplete JPEG during initial training.
The complete archive repaired it; all extracted images were subsequently size/CRC verified.
Download/extraction now uses atomic image writes. macOS resource-fork files in the ZIP
are excluded from the dataset loader. Default Poisson trimming produced empty meshes
on the small block; depth 9 with trim 0 generated nonempty meshes, now validated explicitly.
Earlier development runs are retained for audit and should not replace the selected runs.

The PDF calibration is interpreted as Brown distortion with center offsets. Camera
references are estimated by bundle adjustment; trajectory metrics express agreement with
that reference, not independent ground truth. The STL is a GCP network, not a surveyed
scene surface. No dense ground-truth metrics, semantic labels, or performance improvements
have been fabricated. COLMAP bundle adjustment and PatchMatch remain unweighted internally.
