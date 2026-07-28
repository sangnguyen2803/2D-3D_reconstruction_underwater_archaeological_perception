# Semantic model card v2

**Status:** no trained Mermaid six-class student is released. The available `untrained_smoke_v2.pt` and ONNX smoke model have an ImageNet-pretrained MIT-B0 encoder and a randomly initialized six-class decoder. They test software/export only and are rejected by normal semantic experiment inference.

## Intended use and interfaces

Local underwater research and evaluation. The standalone model accepts RGB floats in [0,1], NCHW at 384x512; normalization is inside the wrapper. It returns six full-resolution logit channels. Calibrated inference resizes RGB, divides logits by validation temperature and returns class probabilities. The API accepts an image and returns a class-mask PNG encoded as base64, per-class pixel fractions and an optional dynamic mask. ONNX has dynamic batch and fixed spatial resolution.

SegFormer B0 is the student; frozen DINOv2 linear probe and light decoder are baselines. Training uses train-slab pseudo masks only, AdamW, AMP on CUDA, cosine learning-rate schedule, label smoothing, ignore=255, batch 2 and accumulation 4. Validation hand masks select checkpoints, temperature and the dynamic threshold. Test labels never fit these quantities.

## Current measurements

Segmentation mIoU, per-class IoU, pseudo-label quality against hand labels, calibrated ECE and final Q1-Q5 results are **unavailable** until annotation. Deployment smoke parity and timings are recorded in `outputs/semantic_onnx_smoke_v2/deployment_v2.json`. Those timings use an untrained head, batch 1, 384x512, five warmups and twenty repetitions on the development machine. They exclude decoding, resizing, transfers and API overhead; they are not trained-model deployment benchmark results.

## Provenance and limits

Checkpoints record model configuration, frozen backbone revision, label-manifest hash, criterion hash, seed, fitting image IDs and validation calibration. Downstream class-aware checkpoints bind to the segmentation checkpoint SHA-256 and feature-spec version. Cache identity includes image state, checkpoint hash, temperature, input/output resolution and runtime backend.

Underwater color casts, ambiguous rock/attached organisms, small artifacts, missed fish, rare-class pseudo-label failure and class/location confounding are known risks. Predicted confidence is an experimental reference-reprojection proxy, not surveyed geometric truth. Split-conformal intervals assume exchangeability and may lose coverage under spatial distribution shift. Multiple observations of one track are grouped for calibration/bootstrap.

The original v1 16-image region is development data already examined. Three descriptive sub-blocks of its larger test slab do not substitute for three separately retrained spatial folds.

See the [semantic dataset card](dataset_semantic_v2.md) for model/data licenses. The student is intended for research/evaluation under the source model's terms. Image-derived masks and checkpoints are kept locally; Docker packaging does not change their license.

Release conditions: human Gate 0 passed; criteria and manifests frozen; completed seed/fold benchmark on hand masks; trained-checkpoint ONNX parity/timing measured; known failures reported. None of the missing quality measurements should be filled with smoke-test numbers.
