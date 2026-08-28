# Fifteen-commit publication and review record

The repository owner supplied all fifteen Git commit subjects before publication. Subjects are preserved exactly, including capitalization and spelling, without numeric prefixes. Each batch is reviewed, staged, validated from a cumulative isolated export, committed and pushed in order. Existing history is not rewritten.

The first commit introduces a package scaffold. The full experiment pipeline becomes runnable after the ablation commit; documentation, analysis notebooks and CI follow in the final commit.

GitHub contains source, configurations, tests, documentation and notebook inputs. Dataset archives/images, pretrained weights, CUDA binaries, caches, environments, MLflow runs and experiment outputs remain local. Executed notebook originals are preserved at `outputs/publication/executed_notebooks/`; published notebook outputs are cleared.

Each Python batch passed lint, formatting and compilation checks before publication. The final full suite contains 26 tests, including four CLI regression cases, including three downloader regression tests added during review. Scientific results in PROJECT_STATUS.md describe the existing 16-image development comparison. Publication smoke checks do not extend that comparison to the entire dataset.

## 01: environmental setup with package structures, license

Commit: [ef24030](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/ef24030431280ca8284b50d7eabb13675c6e4c9e) Files introduced: 6.

Establish Python 3.11 packaging, dependencies, licensing and line-ending rules.

Review: The previous unanchored data ignore rule hid source/configuration files. Root-only exclusions now keep downloads local while publishing the Mermaid loader and Hydra configuration.

Validation: An isolated scaffold built a wheel successfully; wheel contents and dependency syntax were verified.

## 02: image input/output setup

Commit: [87ecd12](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/87ecd12f36a0bebcf3ac3982d85c15b1b2be763e) Files introduced: 1.

Provide RGB image I/O, JSON artifacts, cache fingerprints and atomic NPZ writes.

Review: Unicode filenames work; RGB channel order survives a PNG round trip. JSON rejects non-finite numbers, and temporary NPZ files are removed after replacement.

Validation: Isolated smoke checks covered Unicode RGB I/O, setting-sensitive fingerprints, strict JSON and atomic cache writes.

## 03: download and load epita mermaid datasets

Commit: [ab03f97](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/ab03f971f67e6af090ab43a063719636ee24c9f4) Files introduced: 8.

Download official Mermaid assets and associate images with published reference poses.

Review: Fixed explicit directory entries during ZIP extraction and finalization of an already-complete partial download. Extraction validates paths and CRCs. Metadata and image selection remain reproducible.

Validation: Three offline download regression tests passed: directory entries, archive traversal rejection and completed partial-file resume. Loader formatting passed. The real 1,244-image inventory was checked during pipeline review.

## 04: preprocessing and testing data

Commit: [c94bd39](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/c94bd39877a5b7a3ea142ec04b33d6164496b876) Files introduced: 3.

Resize development images and validate pose loading, dataset filtering and trajectory splits.

Review: Exact per-axis resize scales account for rounded dimensions. The loader excludes macOS metadata and preserves missing pose references; split boundaries discard nearby frames.

Validation: Four focused data tests passed, including 3840 x 2880 to 960 x 720 scaling and resource-fork exclusion.

## 05: config rootSIFT for feature matching

Commit: [318382d](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/318382d4c8cbd998a104666fcf4527c4ab14ffd2) Files introduced: 3.

Extract RootSIFT and match descriptors with mutual nearest neighbors and ratio filtering.

Review: Empty images return correctly shaped feature arrays. RootSIFT vectors are nonnegative and normalized; mutual matches are unique. Reliability reduces correspondence scores.

Validation: Two matching tests passed; a smoke check covered blank-image extraction and descriptor normalization.

## 06: geometric verificaion with ransac

Commit: [78134b4](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/78134b465a01efb1c0d511587f1b909b286a3249) Files introduced: 2.

Verify correspondence geometry and refine a confidence-weighted fundamental matrix.

Review: Normalized eight-point fitting enforces rank two. Confidence influences sampling and inlier support; this adapter does not change COLMAP bundle adjustment.

Validation: Three geometry tests passed: exact synthetic geometry, rejection of low-confidence outliers and fewer than eight matches.

## 07: add metric eval

Commit: [989bb67](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/989bb6736ca22d00109bf0ef9de6f6b4c29cfcaa) Files introduced: 3.

Measure Sim(3)-aligned trajectory agreement and support future independent dense-surface evaluation.

Review: Monocular gauge freedom is handled explicitly. Published BA poses are reference estimates; dense metrics require an independently measured, aligned surface.

Validation: Three evaluation tests passed: scale/rotation/translation recovery, trajectory gauge invariance and identical/empty dense clouds.

## 08: colmap sparse reconstruction adapter

Commit: [6040ff7](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/6040ff792abfcf8e199cdc07e4d0ceac61575dd0) Files introduced: 5.

Import keypoints and filtered matches into COLMAP, map a sparse model and export poses/points.

Review: The database uses the PyCOLMAP schema and adds the OpenCV-to-COLMAP half-pixel offset. No neural descriptors are disguised as uint8 SIFT. Calibration interpretation remains documented.

Validation: The database round-trip test passed, checking coordinates and retained correspondence indices. Sparse reconstruction is not rerun during publication.

## 09: add setup frozen DINOv2 files

Commit: [8dfd108](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/8dfd1086df47895aab5b397c3af2aa10cd40d6d7) Files introduced: 2.

Set up a frozen official DINOv2 backbone and interpolate patch tokens at SIFT locations.

Review: The download entry point pins an upstream revision. Padding preserves image coordinates, and inference disables gradients. This is a hybrid descriptor setup with SIFT localization.

Validation: An isolated smoke check used cached official S/14 weights: frozen parameters, padded non-multiple-of-14 image sizes, finite normalized descriptors and empty keypoint handling all passed.

## 10: update cross-view consistency

Commit: [af4d09e](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/af4d09e8335e05adc7cb312f61a7641927b84cea) Files introduced: 3.

Estimate local image reliability and aggregate weak cross-view correspondence evidence.

Review: Fixed cue scales avoid making uniformly poor images look reliable. Beta smoothing gives unobserved features a neutral repeatability prior; reference evaluation poses are not used as labels.

Validation: Four visibility tests passed: blur sensitivity, flat-region behavior, boundary sampling and repeatability updates.

## 11: edit training files

Commit: [463704d](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/463704d63458be268cd14347fabc4177434ff428) Files introduced: 3.

Train and load a small reliability MLP using frozen features and geometric weak labels.

Review: Training separates validation by contiguous frame groups. Checkpoints retain backbone/revision and training-image provenance. Soft-label BCE is not independent confidence calibration.

Validation: The checkpoint round-trip and bounded-score test passed. The later pipeline smoke check also verified disjoint acquisition lists and a greater-than-20-frame gap.

## 12: setup dense mvs, point confidence

Commit: [266ec13](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/266ec13b48641f5f4027558d38fd978d2174b086) Files introduced: 7.

Run external CUDA COLMAP MVS and attach sparse/dense observation confidence.

Review: MVS uses geometric consistency, validates nonempty outputs and records logs. Confidence is heuristic evidence; Poisson trim=0 can close unsupported holes. BA and PatchMatch remain internally unweighted.

Validation: The COLMAP binary-depth layout test passed. This publication review does not rerun CUDA MVS; earlier measured dense outputs are recorded in PROJECT_STATUS.md. A local pytest cache-write warning did not affect the test result.

## 13: add plots for visualization

Commit: [fb2d3bb](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/fb2d3bbf2707c55adaf9438964319227071f1ee2) Files introduced: 3.

Export scientific figures headlessly and provide a local point-cloud viewer.

Review: Fixed degenerate trajectory plotting so failed Sim(3) alignment does not turn visualization into a reconstruction failure. All figure labels distinguish heuristic confidence and reference agreement.

Validation: Four PNG figure exports were validated; degenerate trajectories were skipped safely. The desktop viewer is opened only by its explicit CLI command.

## 14: perform ablations

Commit: [8e7f023](https://github.com/sangnguyen2803/2D-3D_reconstruction_underwater_archaeological_perception/commit/8e7f02370e5b3a834d7f99dd59d041821c5bd3f5) Files introduced: 20.

Connect the modules with Hydra experiments, immutable runs, caches and controlled ablations.

Review: All five variants compose correctly. Pair cache keys include semantic fusion weight; comparisons require identical ordered inputs. Learned inference checks backbone provenance and temporal train/evaluation separation.

Validation: An isolated two-image Mermaid matching run produced 141 candidates and 140 inliers at the review settings. A second run reproduced pairs/inliers from cache; existing output names were rejected. The full dataset inventory and train/eval lists were also checked. These are smoke checks, not a full-scene benchmark.

## 15: documenting

Commit: Final documentation commit; its SHA is available in Git history. Files in this batch: 20 (including the entry-point correction).

Publish the English README/status/report, original specification, notebook sources and GitHub Actions checks.

Review: Executed notebook originals are preserved locally before clearing outputs. Notebook image selection now uses the dataset loader, avoiding hidden metadata. Final CLI checks exposed a Hydra configuration path issue in imported wrappers; the entry point now resolves its configuration directory from its file location. Four regression cases verify all wrappers from another working directory. Reports distinguish local artifacts, development results and remaining research work.

Validation: Final validation covers the full 26-test suite, Ruff/Black, notebook schemas and clean outputs, notebook generation, report generation, offline HTML generation, CLI entry points and complete-package wheel building. Remote Actions results are reported separately from local validation.
