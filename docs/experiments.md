# Experiment protocol

| ID | Local descriptor | DINO | Reliability | Geometry |
| --- | --- | --- | --- | --- |
| A | RootSIFT | none | none | unweighted RANSAC |
| B | RootSIFT | B/14 fused | none | unweighted RANSAC |
| C | RootSIFT | B/14 fused | handcrafted | weighted RANSAC |
| C2 | RootSIFT | B/14 fused | handcrafted + consistency | weighted RANSAC |
| D | RootSIFT | B/14 fused | learned MLP | weighted RANSAC |
| S | RootSIFT | S/14 fused | handcrafted | weighted RANSAC |
| B0 | SIFT locations | B/14 only | none | unweighted RANSAC |

Each controlled comparison uses the same immutable image list, resolution, seed,
calibration, pair window, ratio test, and mapper settings. Feature/pair caches include
source image state, configuration, backbone revision, calibration, and checkpoint hash.
Output directories cannot be overwritten. Compare a minimum of three random seeds
and multiple trajectory blocks before making a general improvement claim.

The development evaluation block uses acquisition IDs 0,2,...,30. The learned head's
development training block uses 200,202,...,262. These blocks are temporally separated;
their spatial independence has not been established. The MLP internally holds out a
contiguous part of its training block, with a frame-group gap. These are development
experiments, not a final locked benchmark on the entire scene.

## Measurements

2D: candidate matches, geometric inliers, inlier ratio, pair success, and median
Sampson distance on accepted inliers. The current logs do not measure ground-truth
correspondence precision; self-verified inlier ratio can favor restrictive filtering.

Sparse: registered camera count, point count, mean reprojection error, and mean track
length. Count drops indicate potential loss of completeness but are not surface
completeness measurements.

Trajectory: Sim(3)-aligned center error/ATE, camera rotation error, and successive
relative-pose translation/rotation errors, against published bundle-adjustment poses.
Alignment estimates scale/rotation/translation from the evaluated trajectory itself,
which is standard monocular gauge handling but not an independent metric scale test.

Dense: count of fused points, depth-supported observations, confidence statistics,
and visual comparison. Dense surface accuracy, completeness, Chamfer distance, and
F-score must remain unreported until an aligned, independently surveyed surface is
available. `evaluation.metrics.dense_metrics` supports that future reference input.

## Sensitivity and failure analysis

Sweep reliability thresholds, semantic weight, quality scales, RANSAC pixel threshold,
and pair windows on development blocks. Lock chosen values before held-out evaluation.
Inspect failed pairs, disconnected components, repeated textures, vegetation, low
baseline, and low-texture seabed. Report reductions in point support even when
reprojection error improves. Runtime includes cache reuse and therefore is not a fair
backbone comparison unless caches are cold and timing stages are separated.
