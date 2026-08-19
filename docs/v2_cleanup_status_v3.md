# Obsolete v2 artifact cleanup status

The owner requested removal of unnecessary v2 folders and results on 8 October 2026. The following twelve directories were inspected and resolved inside the repository before attempting removal:

- `outputs/comparison_v2`
- `outputs/reliability_training_v2`
- `outputs/semantic_ablation_plan_v2`
- `outputs/semantic_audit_v2`
- `outputs/semantic_deployment_smoke_v2`
- `outputs/semantic_labels_v2`
- `outputs/semantic_onnx_smoke_v2`
- `outputs/semantic_suggestions_v2`
- `outputs/v1_compatibility_v2`
- `docs/figures_v2`
- `cache/wheels_final_v2`
- `cache/wheels_v2`

**Status: removal was not performed.** The execution approval system rejected the PowerShell deletion command with `blocked by policy`. No directory was removed and the old review server was not stopped by that command. The inspected inventory and failure record are saved in `outputs/segformer_folds_v3/v2_cleanup_inventory_v3.json` and `v2_cleanup_status_v3.json`.

The v3 three-fold benchmark continues independently. Baseline, v1, v3, datasets and model weights were excluded from this cleanup. In particular, the baseline pure-DINO ablation `semantic_gpu_eval` is historical baseline evidence, not a v2 result.

The `src/underwater_vision/semantic` package cannot simply be removed: v3 label construction imports `semantic.protocol.spatial_folds`, and v3 point confidence imports generic regression/conformal helpers from `semantic.confidence`. Optional feature, research and serving adapters also reference this package. Removing all v2 code would require moving shared helpers and updating these imports first. This artifact cleanup did not modify those dependencies.
