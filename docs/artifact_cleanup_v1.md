# Artifact cleanup v1

The owner requested removal of unnecessary checkpoints and large files. This
cleanup removed **714 files**, freeing **7,132,895,533 bytes** (7.13 GB, 6.64 GiB).
It preserved all **64 model/checkpoint artifacts**, verified against their
pre-deletion SHA-256 hashes.

| Removed artifact | Reason |
| --- | --- |
| `data/raw/mermaid/107179.zip` | Redundant download: all 1,244 extracted source photographs passed ZIP size and CRC verification. |
| `tools/colmap-x64-windows-cuda.zip` | Redundant download: all 294 extracted COLMAP files passed ZIP size and CRC verification. |
| 712 NPZ files in `cache/matchability_dino_eval_v3/` | Obsolete evaluation cache: current code uses the lossless `matchability_dino_eval_fp32_v3/` cache. |

The extracted source images, reference poses, installed COLMAP executable,
current feature caches, nine SegFormer checkpoints, fitted baselines, ONNX
deployment model, saved predictions, meshes and historical reports remain
available. No current checkpoint was redundant enough to justify removal.

Every deletion used an inspected literal file path and was nonrecursive.
Evidence is saved in `outputs/artifact_cleanup_v1/`: the deletion plan,
archive verification counts, preserved-model hashes, per-file deletion status
and final verification. Historical download manifests still describe the
original acquisition; a missing ZIP is expected after this cleanup.
