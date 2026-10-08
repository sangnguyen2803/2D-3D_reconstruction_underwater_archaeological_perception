# Visibility-Aware Underwater 2Dâ€“3D Reconstruction

A research pipeline for Mermaid underwater imagery: frozen DINOv2 patch features,
local RootSIFT descriptors, visibility-weighted correspondence, geometric verification,
COLMAP SfM/MVS, and observation-based 3D confidence.

The hypothesis is that local reliability can improve geometric reconstruction. Improvements
are **not assumed**. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the measured state,
artifacts, dataset inventory, and remaining research work.

The GitHub repository contains code, configurations, tests, documentation and notebooks.
Downloaded data, model weights, binaries and experiment outputs remain in the prepared
local workspace. A fresh clone needs the download and experiment commands below.
See the [commit review record](docs/commit_plan.md) for the fifteen publication batches.

## Setup (Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[neural,tracking,visualization,dev]"
# NVIDIA GPU: install the CUDA build appropriate for your driver.
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

The prepared local environment uses system packages to avoid duplicating the existing
scientific stack. For an independent install, create a fresh environment as above.
`environment.yml` offers a Conda alternative. `requirements-lock.txt` records the
versions used during local verification; CUDA dependencies are platform-specific.

## Official data

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --workers 6
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
```

The downloader discovers assets from the official SEANOE dataset page, validates file
sizes, records local SHA-256 hashes, retries/resumes HTTP ranges, checks ZIP CRCs, and
protects against archive path traversal. Local hashes are integrity records, not
publisher-provided checksums. A smaller real-image download is available with
`scripts/download_sample.py --count 24 --stride 2`.

Dataset: [EPITA/LRE description](https://www.lre.epita.fr/productions/mermaid/),
[SEANOE DOI 10.17882/97987](https://doi.org/10.17882/97987).
Keep the original metadata and cite Avanthey & Beaudoin (2023). The dataset license is
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/);
the repository's code license does not replace the data/model licenses.

The archive/PDF and the EPITA overview disagree on image count, resolution, and size.
`inspect_dataset.py` reports the actual files. Published camera poses come from bundle
adjustment. The reference STL represents the GCP network, **not dense ground-truth
surface geometry**. Accordingly, trajectory results are reference agreement and dense
Chamfer/F-score results are unavailable without an additional surface survey.

## Reproducible experiments

Freeze acquisition IDs before comparing methods:

```powershell
.\.venv\Scripts\python.exe scripts/make_selection.py --count 16 --stride 2 --output configs/evaluation_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=classical data.image_list=configs/evaluation_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=neural data.image_list=configs/evaluation_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=visibility_aware data.image_list=configs/evaluation_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=consistency data.image_list=configs/evaluation_images.txt
```

All commands use Hydra: `data.max_size=960`, `matching.window=6`, `seed=42`,
`model.backbone=dinov2_vits14`, `stage=features`, and `stage=matches` are supported.
Every run has an immutable output directory, resolved YAML, input/cache manifest,
pair statistics, status/error record, poses, point cloud, confidence, and figures.
Existing run names are rejected. Add `tracking=true` for local MLflow tracking.

The neural baseline uses DINOv2 **with local keypoint localization**. Frozen patch
descriptors are interpolated at SIFT coordinates and fused with RootSIFT for matching;
DINOv2 is not presented as a complete feature matcher. Set
`model.semantic_weight=1.0` for a DINO-only descriptor ablation at SIFT locations.

## Learned reliability

Use a separate acquisition block. Do not train on the evaluation images:

```powershell
.\.venv\Scripts\python.exe scripts/make_selection.py --start-id 200 --count 32 --stride 2 --output configs/training_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=neural data.image_list=configs/training_images.txt stage=matches run_name=reliability_training
.\.venv\Scripts\python.exe scripts/train_reliability.py --run outputs/reliability_training
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=learned data.image_list=configs/evaluation_images.txt matching.learned_checkpoint=outputs/models/reliability.pt
```

Training uses geometric inlier frequency on observed candidate matches, with a
contiguous validation block and an excluded frame boundary. The inference pipeline
rejects overlapping training/evaluation acquisition IDs and a 20-frame neighborhood.
Temporal separation is not proof of spatial separation; inspect the trajectories.

## Dense reconstruction

The Windows PyCOLMAP wheel supports CPU SfM. PatchMatch needs a CUDA-enabled COLMAP
executable. The local `tools/colmap` folder contains the official Windows CUDA release
when its download has completed. Locate the executable and pass its absolute path:

```powershell
.\.venv\Scripts\python.exe scripts/run_mvs.py --run outputs/classical_eval --colmap "C:/path/to/colmap.exe"
.\.venv\Scripts\python.exe scripts/dense_confidence.py outputs/classical_eval/dense
```

Or run SfM and MVS together with `reconstruction.dense=true` and
`reconstruction.colmap_executable=...`. Dense output includes `fused.ply`, depth/normal
maps, a Poisson mesh, and `fused_confidence.ply`. Confidence uses depth-consistent view
support and source-image quality; it is a heuristic, not a calibrated error probability.

## Inspect results

```powershell
.\.venv\Scripts\python.exe scripts/compare_experiments.py outputs/classical_eval outputs/neural_eval outputs/visibility_eval outputs/learned_eval
.\.venv\Scripts\python.exe scripts/view_pointcloud.py outputs/visibility_eval/sparse_confidence.ply --confidence
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m black --check src scripts tests
```

The Open3D viewer opens a desktop window only when you invoke it. Matplotlib exports
are generated headlessly. `scripts/build_demo.py` generates a local HTML artifact
with measured tables, figures, and interactive point clouds; no service deployment
is required. Notebooks explore existing outputs without silently rerunning expensive models.
Published notebooks have cleared outputs. The executed originals are preserved locally
under `outputs/publication/executed_notebooks/`.

For notebooks, register a kernel inside the local environment:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name underwater-vision
```

The checked comparison uses `neural_gpu_eval`, `visibility_gpu_eval`, and
`consistency_gpu_eval`; the earlier CPU development runs are also retained. The final
demo compares `classical_eval` against `visibility_gpu_eval`, with camera-aligned sparse
and dense clouds, camera paths, original RGB, and confidence colors.

Further reading: [methodology](docs/methodology.md), [experiment protocol](docs/experiments.md),
[limitations](docs/limitations.md), [dataset conventions](docs/dataset.md).


## Reference-supervised extension

The reference-supervised extension adds spatially separated match supervision,
global DINO pair retrieval and held-out point-error prediction. The original
pipeline stays the default; select `experiment=learned_matches` to use a trained
filter. See [the extension overview](docs/v1_publication.md) for requirements,
commands and the historical development findings. Results and checkpoints must
be generated or supplied locally.
