# Visibility-Aware Underwater 2D-3D Reconstruction

A modular research pipeline for underwater archaeology imagery, combining
RootSIFT, frozen DINOv2, reference-supervised correspondence filtering and
COLMAP sparse reconstruction, with optional dense MVS and point confidence.

## Dataset

The verified EPITA/LRE Mermaid acquisition contains **1,244 RGB images** and
published reference camera poses. Standard experiments use 960 x 720 inputs.
Download assets from [SEANOE](https://doi.org/10.17882/97987); retain attribution
and respect the dataset's CC BY-NC-ND 4.0 license.

## Quick start

Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[neural,research,dev]"
.\.venv\Scripts\python.exe scripts/download_data.py --workers 6
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
.\.venv\Scripts\python.exe scripts/make_selection.py --count 16 --stride 2 --output configs/evaluation_images.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py experiment=classical data.image_list=configs/evaluation_images.txt
```

Use a new run name for repeated experiments. Frozen DINOv2 variants download
pretrained weights on first use. Dense MVS requires a CUDA-enabled COLMAP
executable configured separately.

## Components

- Local features, global image retrieval and geometric match verification.
- Spatially separated reference supervision, MLP/context match filtering and
  held-out point-confidence evaluation.
- Configurable experiments, cached features, camera/point-cloud exports and
  visualization. Optional MLflow tracking is enabled with `tracking=true`.

Additional SegFormer matchability, calibration and reconstruction experiments
exist in the local workspace and are being published in separate code batches.

## Evaluation and artifacts

Classification AP, calibration and reconstruction accuracy are evaluated
separately. Pose metrics measure agreement with published bundle-adjusted
cameras; independently surveyed dense-surface accuracy is unavailable.
A complete 1,244-image SfM/MVS run has not been demonstrated.

Dataset files, weights and outputs are excluded from Git. Local trained
checkpoints were removed when the project was paused; learned inference needs
restored weights or retraining. Source code and saved numeric results remain.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

See [dataset conventions](docs/dataset.md), [methodology](docs/methodology.md)
and [reference-supervised extension](docs/v1_publication.md) for further details.
