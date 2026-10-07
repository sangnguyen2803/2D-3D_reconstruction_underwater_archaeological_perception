# Methodology

## Hypothesis and causal comparison

Local image evidence and correspondence stability are imperfect indicators of feature
reliability. We test whether their use improves correspondence geometry, registered
camera coverage, sparse track support, and reconstruction quality. The pipeline makes
no novelty claim for a weighted similarity product and no SOTA claim.

## Coordinate handling

Images are resized with preserved aspect ratio, normally to a longest edge of 960.
Intrinsics scale independently along each axis because rounded dimensions may differ.
RootSIFT keypoints use OpenCV coordinates. Import adds 0.5 pixels for COLMAP's pixel
center convention. Fundamental-matrix verification uses undistorted keypoints and
pixel-scaled intrinsics. Raw distorted coordinates enter COLMAP with FULL_OPENCV.

The source calibration table lists focal length, principal-point offsets, and Brown
coefficients. We interpret these as Metashape-style center offsets. The PDF does not
explicitly establish this convention; reference-pose agreement is a sanity check,
not an independent calibration validation. We freeze intrinsics for controlled runs.

## Features and correspondence

RootSIFT normalizes SIFT histograms by their L1 norm and takes their square root.
The frozen official DINOv2 B/14 backbone returns normalized patch tokens. ImageNet
normalization and replicated bottom/right padding preserve pixel coordinates; bilinear
sampling associates patch descriptors with SIFT locations. Concatenating
`sqrt(1-alpha) * RootSIFT` and `sqrt(alpha) * DINO` yields a fused descriptor.
Mutual nearest-neighbor matching and a ratio test produce unique candidate pairs.

Baseline A uses RootSIFT; B uses fused descriptors; C adds image reliability;
D replaces the handcrafted reliability with a learned head. Semantic-only descriptors
and S/14 form additional ablations. Neural descriptors are never quantized to pretend
they are COLMAP uint8 SIFT descriptors. COLMAP needs only imported keypoints and
matches for its subsequent geometric verification and mapping.

## Deterministic reliability

`R(x,y)` combines local contrast, Laplacian variation, gradient texture, quantized
entropy, exposure, and a lightly weighted color balance cue. Fixed saturating scales
avoid making globally poor images appear reliable through per-image normalization.
Weights sum to one and scores are clipped to [0,1]. The color cue has only 0.02 weight;
underwater color imbalance is not assumed to be a physical visibility measurement.
Parameters are hypotheses exposed for sensitivity analysis, not validated constants.

Correspondence confidence combines local and semantic similarity and multiplies the
two source reliability scores. Low-score pairs are filtered. Weighted RANSAC samples
eight-point hypotheses proportionally to confidence and scores weighted inlier support.
A normalized weighted eight-point fit enforces rank two and refines the best model.
The unweighted OpenCV RANSAC solution remains an initial candidate. COLMAP independently
verifies the surviving matches before essential-matrix pose recovery and bundle adjustment.

**COLMAP bundle adjustment is unweighted.** Confidence changes its input graph, not its
internal objective. Do not describe this implementation as weighted bundle adjustment
or visibility-aware PatchMatch; those would require backend changes.

## Cross-view and learned reliability

The cross-view ablation estimates a Beta-smoothed inlier-frequency posterior for each
observed feature, then re-verifies correspondence geometry. Initial RANSAC labels can
be wrong and the second pass can reinforce errors. Reference evaluation poses are not
used to construct these labels.

The learned head is `C -> C/2 -> GELU -> 1 -> sigmoid`. Training uses BCEWithLogitsLoss
on soft geometric inlier frequencies for points observed in at least two pairs.
Unobserved features are not assigned negative labels. Validation uses contiguous frame
groups with a discarded boundary. Checkpoints include backbone/revision and training
image names, allowing inference to reject temporal train/evaluation overlap.

## Confidence propagation

Sparse confidence is the product of saturating view support, reprojection quality,
maximum triangulation-angle quality, and mean source-feature reliability. Dense
confidence separately projects fused points into undistorted source cameras and
requires agreement with geometric MVS depth maps; view support, relative depth residual,
and local image visibility determine the score. Dense confidence is currently
handcrafted even in the learned-matching experiment.

Confidence fields are saved alongside original XYZ/RGB. Their numeric values are
heuristic evidence scores. Calibration against independent reconstruction errors
remains necessary before interpreting them as probabilities.
