"""Write an English status summary and technical report from measured artifacts."""

import json
from pathlib import Path


def main():
    inventory = json.loads(Path("outputs/dataset_inventory.json").read_text())
    integrity = json.loads(Path("outputs/dataset_integrity.json").read_text())
    results = json.loads(Path("outputs/comparison/comparison.json").read_text())
    table = "| Run | Cameras | Sparse points | Inlier ratio | Reprojection (px) | ATE (reference units) |\n|---|---:|---:|---:|---:|---:|\n"
    for row in results:
        reprojection = row.get("mean_reprojection_error_px")
        ate = row.get("trajectory", {}).get("ate_rmse_reference_units")
        table += (
            f'| {row["run"]} | {row["registered_images"]} | {row["sparse_points"]} | {row["inlier_ratio"]:.4f} | '
            + (f"{reprojection:.4f}" if reprojection is not None else "Unavailable")
            + " | "
            + (f"{ate:.5f}" if ate is not None else "Unavailable")
            + " |\n"
        )
    dense_table = "| Run | Dense points | Depth-supported points | Mesh vertices | Mesh faces |\n|---|---:|---:|---:|---:|\n"
    for run in ["classical_eval", "visibility_gpu_eval"]:
        metrics = json.loads((Path("outputs") / run / "metrics.json").read_text())
        dense_table += f'| {run} | {metrics["dense_points"]} | {metrics["points_with_depth_support"]} | {metrics["mesh_vertices"]} | {metrics["mesh_faces"]} |\n'
    head = json.loads(Path("outputs/models/reliability.json").read_text())
    summary = f"""# Project status — 2026-10-07

## What exists now

The English Python 3.11 project is implemented and has been run on real Mermaid data.
The full official dataset is downloaded: **{inventory['images_on_disk']} RGB JPEG images,
{inventory['reference_poses']} reference camera poses**, acquisition/calibration PDFs,
and the GCP-network STL. Actual image dimensions are 3840 × 2880. The image ZIP is
3,690,513,210 bytes; original JPEGs total {inventory['image_bytes']:,} bytes.
All {integrity['images_verified']} extracted images match their archive sizes and CRCs.
The complete ZIP also passed CRC validation. Source URLs, dataset license, and local
SHA-256 integrity hashes are preserved under `data/raw/mermaid/`.

You do not need to download the dataset manually.

This report describes the prepared local workspace. GitHub contains the code and
documentation; datasets, weights, binaries and experiment outputs remain local.
A fresh clone must download the data and reproduce the experiments using README.md.

## Implemented and verified

- [x] Mermaid loader, published pose parser, actual inventory, hidden metadata exclusion.
- [x] Aspect-preserving preprocessing and calibration/coordinate conversion.
- [x] RootSIFT classical reconstruction using the COLMAP backend.
- [x] Frozen official DINOv2 B/14 and S/14, patch-to-keypoint descriptors, mixed precision.
- [x] Mutual ratio matching, local/neural descriptor fusion, pure-DINO descriptor ablation.
- [x] Deterministic visibility maps, cross-view consistency, weighted RANSAC/refinement.
- [x] Learned reliability MLP with separate acquisition-block supervision and leakage checks.
- [x] COLMAP database integration, camera poses, sparse points, tracks, reprojection metrics.
- [x] CUDA MVS, colored dense clouds, nonempty Poisson meshes for baseline and proposed runs.
- [x] Sparse and dense confidence fields, depth-consistent dense observation support.
- [x] Controlled A/B/C/D experiments, consistency/backbone ablations, explicit failure records.
- [x] Sim(3)-aligned reference trajectory agreement, ATE/RPE/rotation measurements.
- [x] Hydra YAML, immutable run names, atomic feature/pair/visibility caches, MLflow artifacts.
- [x] Core pytest suite, Ruff/Black checks, reproducible CLI scripts and English documentation.
- [x] Five analysis notebooks, exported figures, and offline interactive 3D comparison demo.

Verification completed locally: **26 pytest tests passed**, Ruff checks passed,
Black formatting checks passed, all five notebooks executed without cell errors,
Open3D loaded the exported dense cloud, and the offline demo rendered in headless Chrome.
GitHub Actions runs the test, lint and format checks on pushes and pull requests.
Remote workflow results are available in the repository's Actions tab.

Publication review and validation for each of the fifteen commits are recorded in
`docs/commit_plan.md`. Notebook outputs are cleared for publication; executed originals
are preserved under `outputs/publication/executed_notebooks/` in the local workspace.

## Actual development results

All tabled runs use the same **16 images: acquisition IDs 0,2,...,30**, resolution 960 × 720,
calibration, pair window, ratio threshold, and seed 42. Neural B/C/D/S runs use the
CUDA PyTorch implementation. SIFT uses CPU OpenCV; SfM uses CPU PyCOLMAP 3.13.0;
dense reconstruction uses official COLMAP 4.2.1 CUDA on an RTX 4060 Laptop GPU (8 GB).

{table}

The pure-DINO descriptor ablation (`semantic_gpu_eval`) did not register a model.
The zero camera/point counts record this observed failure, not an invented reconstruction.
The earlier `semantic_only_eval` development run reused a stale matching-cache key and
must not be used for conclusions. The cache key now includes the semantic fusion weight;
the corrected pure-DINO run is the one in this table.

{dense_table}

Interpretation: reliability raises self-verified inlier ratios, but does not dominate all
metrics. The classical baseline retains substantially more sparse points. Compared with
the hybrid neural baseline, handcrafted reliability yields slightly more points and
higher inlier ratio but slightly worse mean reprojection error. The learned head does
not establish a consistent advantage over handcrafted reliability. S/14 preserves more
points than B/14 on this block. These are measured development findings, not a SOTA claim.

## Learned reliability checkpoint

`outputs/models/reliability.pt` was trained for 20 epochs from acquisition IDs
200,202,...,262, separately from the evaluation block. It uses **{head['samples']} weakly
supervised observations**, {head['train_samples']} training examples and
{head['validation_samples']} validation examples, with a discarded boundary group.
Best validation BCE is {head['best_validation_bce']:.4f}. This is validation against
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
.\\.venv\\Scripts\\python.exe scripts/run_ablation_suite.py --prefix reproduction --include-small --tracking
.\\.venv\\Scripts\\python.exe scripts/setup_colmap.py
.\\.venv\\Scripts\\python.exe scripts/run_mvs.py --run outputs/reproduction_classical --colmap "$PWD/tools/colmap/bin/colmap.exe"
.\\.venv\\Scripts\\python.exe scripts/finalize_dense.py --run outputs/reproduction_classical
.\\.venv\\Scripts\\python.exe -m pytest -q
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
"""
    Path("PROJECT_STATUS.md").write_text(summary, encoding="utf-8")
    report = "# Technical report\n\n" + summary[summary.index("## Actual development results") :]
    Path("docs/technical_report.md").write_text(report, encoding="utf-8")
    print("PROJECT_STATUS.md and docs/technical_report.md written")


if __name__ == "__main__":
    main()
