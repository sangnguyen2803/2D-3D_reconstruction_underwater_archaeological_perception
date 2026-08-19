# Project Stop: Checkpoint Cleanup v1

Completed on 9 October 2026 at the owner's explicit request to stop the project and delete its checkpoints.

Deleted **70 model/checkpoint files**, totaling **166,892,833 bytes (166.89 MB / 159.16 MiB)**.

Removed project-local PyTorch weights, fitted baseline joblib artifacts and the ONNX model export. Shared pretrained model caches outside the project remain. Every deletion used an explicitly enumerated literal file path; no recursive directory deletion was performed.

Code, configurations, dataset, reference geometry, cached features/predictions, calibration metadata, metrics, figures, sparse models, dense clouds, meshes and historical reports remain. Inference with deleted models requires retraining or restoring the corresponding weights. Historical model paths and hashes remain as provenance and no longer indicate available files.

Earlier summaries remain historical snapshots and were not overwritten. No commit or push was performed.

Verified **60921** inventoried non-checkpoint artifact files with unchanged sizes; **0** checkpoint files remain in the scanned project artifact directories. Preserved content hashes were not recomputed.

Evidence: `outputs/project_stop_checkpoint_cleanup_v1/deletion_plan_v1.json`, `deletion_status_v1.json`, `preserved_artifact_inventory_v1.json` and `verification_v1.json`. The deletion plan records pre-deletion SHA-256 hashes.

The initial combined cleanup command was rejected by automatic approval review with "blocked by policy" before execution. The operation succeeded after separating inventory from deletion and using explicitly enumerated literal paths.

| Deleted artifact | Bytes |
|---|---:|
| `outputs/conditional_precision_v1/context_v1.pt` | 221429 |
| `outputs/conditional_precision_v1/mlp_v1.pt` | 50367 |
| `outputs/geometry_context_expanded_v1/ranker_v1.pt` | 269406 |
| `outputs/geometry_ranker_expanded_v1/ranker_v1.pt` | 98729 |
| `outputs/geometry_ranker_v1/ranker_v1.pt` | 88169 |
| `outputs/matchability_deployment_preview_v3/matchability_v3.onnx` | 15173258 |
| `outputs/matchability_suite_v3/baselines_fold0_s42_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/matchability_suite_v3/baselines_fold0_s42_v3/scale_response_v3.joblib` | 1393 |
| `outputs/matchability_suite_v3/baselines_fold0_s42_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/matchability_suite_v3/baselines_fold0_s42_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/matchability_suite_v3/baselines_fold0_s43_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/matchability_suite_v3/baselines_fold0_s43_v3/scale_response_v3.joblib` | 1393 |
| `outputs/matchability_suite_v3/baselines_fold0_s43_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/matchability_suite_v3/baselines_fold0_s43_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/matchability_suite_v3/confidence_fold0_s42_v3/matchability_10_v3.pt` | 58841 |
| `outputs/matchability_suite_v3/confidence_fold0_s42_v3/robust_8_v3.pt` | 58245 |
| `outputs/matchability_suite_v3/confidence_fold0_s42_v3/v1_gaussian_8_v3.pt` | 58305 |
| `outputs/matchability_suite_v3/confidence_fold0_s43_v3/matchability_10_v3.pt` | 58841 |
| `outputs/matchability_suite_v3/confidence_fold0_s43_v3/robust_8_v3.pt` | 58245 |
| `outputs/matchability_suite_v3/confidence_fold0_s43_v3/v1_gaussian_8_v3.pt` | 58305 |
| `outputs/matchability_suite_v3/filters_fold0_s42_v3/context_12_v3.pt` | 258810 |
| `outputs/matchability_suite_v3/filters_fold0_s42_v3/context_16_v3.pt` | 260090 |
| `outputs/matchability_suite_v3/filters_fold0_s42_v3/mlp_12_v3.pt` | 87849 |
| `outputs/matchability_suite_v3/filters_fold0_s42_v3/mlp_16_v3.pt` | 89129 |
| `outputs/matchability_suite_v3/filters_fold0_s43_v3/context_12_v3.pt` | 258810 |
| `outputs/matchability_suite_v3/filters_fold0_s43_v3/context_16_v3.pt` | 260090 |
| `outputs/matchability_suite_v3/filters_fold0_s43_v3/mlp_12_v3.pt` | 87849 |
| `outputs/matchability_suite_v3/filters_fold0_s43_v3/mlp_16_v3.pt` | 89129 |
| `outputs/matchability_suite_v3/segformer_b0_fold0_s42_v3/model_v3.pt` | 14985779 |
| `outputs/matchability_suite_v3/segformer_b0_fold0_s43_v3/model_v3.pt` | 14985779 |
| `outputs/match_filter_v1/context_v1.pt` | 258805 |
| `outputs/match_filter_v1/mlp_v1.pt` | 87487 |
| `outputs/models/reliability.pt` | 1186501 |
| `outputs/observability_ranker_v1/context_v1.pt` | 221493 |
| `outputs/point_confidence_v1/error_model_v1.pt` | 58345 |
| `outputs/segformer_folds_v3/baselines_fold0_s44_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold0_s44_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold0_s44_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold0_s44_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold1_s42_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold1_s42_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold1_s42_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold1_s42_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold1_s43_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold1_s43_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold1_s43_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold1_s43_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold1_s44_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold1_s44_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold1_s44_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold1_s44_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold2_s42_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold2_s42_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold2_s42_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold2_s42_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold2_s43_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold2_s43_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold2_s43_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold2_s43_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/baselines_fold2_s44_v3/dino_reliability_v3.pt` | 1231509 |
| `outputs/segformer_folds_v3/baselines_fold2_s44_v3/scale_response_v3.joblib` | 1393 |
| `outputs/segformer_folds_v3/baselines_fold2_s44_v3/tabular_boosting_v3.joblib` | 161560 |
| `outputs/segformer_folds_v3/baselines_fold2_s44_v3/tabular_logistic_v3.joblib` | 1521 |
| `outputs/segformer_folds_v3/segformer_b0_fold0_s44_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold1_s42_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold1_s43_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold1_s44_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold2_s42_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold2_s43_v3/model_v3.pt` | 14985843 |
| `outputs/segformer_folds_v3/segformer_b0_fold2_s44_v3/model_v3.pt` | 14985843 |
