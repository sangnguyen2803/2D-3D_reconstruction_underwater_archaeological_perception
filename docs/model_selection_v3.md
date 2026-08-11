# Primary matchability model selection v3

On 8 October 2026, the user selected **SegFormer-B0** as the primary dense matchability backbone. This agrees with the existing primary model in the frozen v3 criteria. The training CLI already defaults to `configs/matchability/model/segformer_b0.yaml`, and the matchability experiment already references its trained fold0 seed42 checkpoint. The decision is recorded in `configs/matchability/selection_v3.yaml`; that file records the selection and existing artifact, rather than changing the frozen criteria or restarting training.

Input: RGB image. Output: a one-channel dense matchability logit/score map, sampled at the original RootSIFT keypoint coordinates. This is not a semantic class mask. Optional match-filter, keypoint-selection, reliability and point-confidence integrations remain separately configurable.

The existing checkpoint is `outputs/matchability_suite_v3/segformer_b0_fold0_s42_v3/model_v3.pt`, SHA256 `0d15f3458008248c24677044a7d51de914561ef6145db184959d21a7f48fce7d`. Its ONNX export is in `outputs/matchability_deployment_preview_v3`. It is an existing fold0/seed42 exemplar, not a final model trained on the entire dataset. Spatial fit/evaluation guards still apply. Do not use its predictions on training/validation views as held-out evidence.

The two completed fold0 seeds achieved AP 0.884234 and 0.882831, versus 0.879086 and 0.880411 for the retrained frozen-DINO MLP control. Three-fold evaluation is incomplete, so this small observed advantage is provisional. Post-temperature ECE was 0.047924 and 0.057485; the second exceeds 0.05. Reports should currently describe outputs as matchability scores rather than universally calibrated probabilities. The added matchability inputs have not demonstrated a downstream benefit: the context filter was worse on the completed fold0 comparisons, and the point heuristic still had lower AUSE.

ResNet18-UNet and frozen DINO+conv implementations and results remain available for optional comparisons. DINOv2 descriptors used for retrieval and correspondence features remain independent components; selecting SegFormer as the dense predictor does not remove them.

The experiment suite remains stopped at the user's request. No new training, reconstruction, Docker launch, commit or push was performed for this model-selection change. See [the stopped progress report](../PROJECT_PROGRESS_V3.md) for outstanding experiments and software verification.
