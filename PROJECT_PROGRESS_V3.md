# Project v3 progress report — stopped for review

Recorded on 8 October 2026, Europe/Paris. The user requested that work stop and that the current state be reviewed. All active v3 suite/training processes were stopped. This is a progress report, not a claim that the v3 specification is complete. No further benchmark is started by this report.

## What has been implemented

The project now has a standalone `matchability` package for dense, one-channel image prediction with sparse supervision at original RootSIFT coordinates. It includes reference-geometry labels, pair eligibility, stronger track-based labels, spatial datasets, coordinate-aware crops and sampling, sparse BCE, pretrained ResNet18-UNet / SegFormer-B0 / frozen DINOv2 heads, seeded training, temperature scaling, inference, caching and statistical baselines.

Four integrations were added as optional components: four matchability inputs added to the original twelve-input match filter; exact top-N keypoint selection with original index provenance; an independent reliability-source switch; and two track-level inputs added to the point-error learner. Feature versions and model hashes are checked to reject incompatible checkpoints. These components can be disabled or used independently. Code integration is implemented; complete experimental validation of every integration is still pending.

New scripts support immutable experiment directories, label audits, training, fold/seed suites, filter and point-confidence evaluation, ONNX export, FastAPI serving and measured-report generation. The v2 semantic package remains separate and is not used in v3 supervision or results. Existing v1/v2 reports and summaries were preserved.

## Data and labels: completed

The dataset was already available locally: 1,272 source images, including 1,244 with published reference poses. No additional download was required for this phase. All 1,244 posed images were processed at 960×720 without CLAHE, with original-coordinate RootSIFT features and lossless RGB PNG inputs.

There are 671,722 detected keypoints: minimum 1, median 218, maximum 4,097. Requesting 4,096 SIFT features does not ensure that many detections. This is a material limitation of the requested 512/1,024/2,048/4,096 exact-budget experiments.

L1 and soft labels were built from bidirectional top-1 candidates and reference-F error. Pairs require at least 15 reference-compatible directed candidates. Positives require at least one candidate at or below 1.5 px; negatives require at least two eligible candidates, all at or above 4 px; everything else is ignored. E0 includes nine threshold settings, error histograms, label stability and eligibility effects for sequential and hybrid graphs. Criteria and the four-epoch initial training budget were frozen after this audit.

The original v1 split is 226 train / 65 validation / 743 test / 210 excluded views. Three rotated spatial protocols use fixed 377/336/376-view blocks plus 155 excluded views and the same approximately 0.4472-unit exclusion gap.

Stronger L2 labels were also completed. They certify L1 positives through mutual, conflict-free reference-camera tracks and held-out reprojection error; uncertified positives are ignored. L1 negatives are retained. Training the L2/soft ablations has not yet happened.

Artifacts: `outputs/matchability_labels_v3`, `outputs/matchability_l2_labels_v3`, and `configs/matchability/criteria_v3.yaml`.

## Benchmark state at interruption

The running suite reached command 14 of 57. Thirteen result directories contain evaluation files: seven dense-network runs, two baseline runs, two match-filter runs and two point-confidence runs. All belong to fold 0. SegFormer fold0 seed44 was interrupted and contains only its resolved configuration; it has no completed model/evaluation artifact. Fold1 and fold2 have not been run.

One of the thirteen recorded runs, DINO-head fold0 seed43, used an early reduced-precision evaluation cache that subsequently failed a strict parity audit. Its artifacts are retained for traceability, but it must be rerun before inclusion in the final benchmark. There are therefore twelve completed runs without that known cache qualification, rather than thirteen fully accepted final runs.

The full suite would include 27 dense-network runs, nine baseline runs, nine filter runs and nine point-confidence runs, followed by aggregation, deployment timing and additional label/SfM experiments. That work is incomplete.

## Initial measured results: same fold, seeds 42 and 43

These are provisional measurements from one spatial region, not three-fold conclusions. AP is higher-is-better; AUSE is lower-is-better.

| Measurement | Seed 42 | Seed 43 | Interpretation |
| --- | --- | --- | --- |
| SegFormer dense test AP | 0.884234 | 0.882831 | Small advantage over the strongest current control; no completed three-seed/fold CI yet |
| DINO + MLP baseline AP | 0.879086 | 0.880411 | Strongest baseline in the two completed baseline runs |
| Tabular boosting AP | 0.847606 | 0.847455 | Stronger than handcrafted and response-only controls |
| ResNet18-UNet AP | 0.829776 | 0.836853 | Below DINO-MLP and SegFormer here |
| SegFormer ECE after temperature scaling | 0.047924 | 0.057485 | Seed43 exceeds the frozen 0.05 calibration bound |
| Retrained 12-input context filter AP | 0.634701 | 0.636315 | Same candidates as the 16-input filter |
| Context filter with four matchability inputs AP | 0.621524 | 0.617832 | Lower than the retrained 12-input control in both seeds |
| Retrained v1 Gaussian 8-input point AUSE | 0.392055 | 0.365613 | Fair baseline on the new track population |
| Robust 8-input point AUSE | 0.347896 | 0.354809 | Controls for loss/calibration changes |
| Matchability 10-input point AUSE | 0.325582 | 0.327160 | Lower than both learned controls by point estimate |
| Heuristic point AUSE | 0.291487 | 0.291487 | Still lower than the ten-input learner |

The third ResNet seed is complete: AP 0.845122, ECE 0.027583. It does not complete the missing third seeds for other models.

For context filtering, paired 95% pair-bootstrap intervals for AP(16) − AP(12) are [-0.026289, -0.000577] at seed42 and [-0.031385, -0.006049] at seed43. These support a negative effect on this evaluated fold/pool, rather than an improvement. They do not establish what will happen on the remaining regions.

For point confidence at seed42, AUSE(10) − AUSE(v1 learned) has CI [-0.126870, -0.017599], and versus robust8 [-0.047927, -0.005772]. The interval versus the heuristic crosses zero: [-0.035235, 0.099425]. At seed43, even the intervals against both learned controls cross zero. Q5 has therefore not been established, and the heuristic remains an essential control.

The original v1 context AP of approximately 0.946 came from a different, compact development candidate pool. The rotated-fold filter test uses 32 views sampled across a slab, including difficult/non-overlapping pairs. Comparing 0.946 directly with the current approximately 0.62 would confound method and population changes. The fair current comparison is twelve versus sixteen inputs on identical candidates.

## Compatibility and deployment already checked

With matchability disabled, the original sixteen-view v1 context reconstruction was rerun: 16 registered cameras, 1,890 sparse points and mean reprojection error 0.4504205128199967 px. The preserved value is 0.4504205128199975 px. Agreement is within 1e-12, providing numerical backward compatibility.

A trained SegFormer fold0 seed42 checkpoint was exported to ONNX. Dynamic batch passed; CPU FP32 maximum absolute logit difference was 0.0000019818544 with atol=1e-4 and rtol=1e-3. A CPU ONNX FastAPI Docker image was actually built and launched. A real image produced a 720×960 16-bit map and top-32 keypoints; health checks and invalid-image HTTP 400 passed. Docker was subsequently stopped to release resources. Formal warmed-up GPU FP32/FP16 latency, throughput and peak-VRAM measurements have not yet been run. The single HTTP smoke-request duration is not a latency benchmark.

Artifacts: `outputs/v1_compatibility_v3`, `outputs/matchability_deployment_preview_v3`, and `outputs/matchability_docker_smoke_v3`.

## Fixes and checks that still need closure

- The initial BF16 DINO evaluation-grid cache failed strict parity with maximum logit error 0.0019002. The replacement stores FP32 values and restores original strides. On an audited real image, cached and fresh grid/logit results had zero difference. Future training uses the lossless cache; the already completed seed43 run still needs replacement under a new immutable name.
- The initial equal-budget random control used the same seed at both image endpoints. Independent per-image seeds and explicit timing warm-up have now been implemented. Earlier budget artifacts remain preserved; the corrected budget reruns are pending.
- The latest complete test run earlier in development passed 84 tests, and a later matchability-only run passed 28 tests. Additional changes and a new cache test were made afterward. The final current-tree full pytest/Ruff/Black verification has not been completed. Those earlier passes must not be presented as final verification of all current files.
- Report/card/figure generation code was added, but the final `PROJECT_SUMMARY_V3.md`, dataset/model cards, figures and central `outputs/comparison_v3/comparison_v3.json` have not been generated. This progress report is the available current summary.
- L1/L2/soft training comparisons, the new-method end-to-end SfM comparisons, complete E5 curves and all remaining folds/seeds are pending. Full 1,244-view COLMAP/MVS reconstruction was not performed in this v3 phase.
- No new Git commit or push was made in this phase. Changes still include earlier v1/v2 work, so the full working-tree diff should not be described as exclusively v3.

## Should work continue?

Yes, if the goal is a defensible research evaluation, but further large benchmark runs should wait for a focused review. The current results do not justify claiming that matchability improves the complete reconstruction pipeline.

First, review label eligibility, calibration behavior, the filter candidate population and the incomplete exact-budget comparisons. Preserve the frozen v3 criteria; changing feature budgets, pair protocols or model tuning after inspecting these tests would require a separately identified exploratory protocol rather than silently redefining v3 success.

Second, close the cache/random-control corrections and run final software checks. This makes the current implementation and comparisons reviewable before spending more GPU time.

Third, complete the remaining spatial folds and seeds, retaining negative outcomes, then run the label ablations, identical-input SfM comparisons and formal deployment timing. Completion is useful even if Q3 or Q5 remains negative: it distinguishes a functioning dense predictor from an independently demonstrated reconstruction benefit.

## Scientific limits

The labels measure RootSIFT/reference-geometry compatibility on one underwater scene. Reference cameras are derived from bundle adjustment, not independent surveyed truth. Epipolar compatibility is necessary but not proof of correct matching. Camera-center separation does not guarantee disjoint visible surfaces. Dense predictions are sparsely supervised at detected keypoints; scores at other pixels or on another scene are unvalidated. Brown calibration does not model underwater refraction. No manual annotation, semantic segmentation labels or v2 semantic results enter this work.

The benchmark is stopped at the user's request. Resumption requires a subsequent user instruction; no background benchmark continuation remains active.
