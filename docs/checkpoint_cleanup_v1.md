# Old checkpoint cleanup v1

Completed on 8 October 2026 at the owner's request. Deleted **8 files**, freeing **182,004,977 bytes** (182.00 MB / 173.57 MiB).

| Deleted file | Bytes |
| --- | ---: |
| `outputs/semantic_deployment_smoke_v2/untrained_smoke_v2.pt` | 14,955,619 |
| `outputs/semantic_onnx_smoke_v2/segmentation_v2.onnx` | 15,174,528 |
| `outputs/matchability_suite_v3/resnet_unet_fold0_s42_v3/model_v3.pt` | 49,962,225 |
| `outputs/matchability_suite_v3/resnet_unet_fold0_s43_v3/model_v3.pt` | 49,962,225 |
| `outputs/matchability_suite_v3/resnet_unet_fold0_s44_v3/model_v3.pt` | 49,962,289 |
| `outputs/matchability_suite_v3/dino_head_fold0_s42_v3/model_v3.pt` | 400,795 |
| `outputs/matchability_suite_v3/dino_head_fold0_s43_v3/model_v3.pt` | 400,795 |
| `mlruns/412626521167535764/4a5bde15cf234d339dc687295c4e6a47/artifacts/model/reliability.pt` | 1,186,501 |

The removed files are two obsolete v2 smoke models, three alternative ResNet-UNet checkpoints, two alternative DINO-head checkpoints, and an MLflow copy byte-identical to the retained `outputs/models/reliability.pt`.

All **64 remaining checkpoint/model files** were verified against their pre-cleanup SHA256 hashes. This preserves all nine SegFormer checkpoints, all four fitted artifacts for each of the nine current baseline runs, the baseline/v1 pipeline models, the partial v3 filter/confidence models and the current v3 ONNX export. Fold0 seed42/43 models remain in the older suite directory because the completed benchmark reuses them.

Numeric results, predictions, reports, dataset and pretrained downloads remain available. The deleted alternative-model weights are no longer available for inference; those backbones can be retrained. Historical reports remain unchanged.

The operation removed only the eight inspected files using nonrecursive PowerShell `Remove-Item -LiteralPath`. All eight are absent, and the nine dense plus nine baseline benchmark artifacts were checked after deletion. No training, code changes, commit or push were performed for this cleanup.

Inventory, pre-deletion hashes and verification: `outputs/checkpoint_cleanup_v1/deletion_plan.json`, `preserved_checkpoints.json`, `deletion_status.json` and `verification.json`.
