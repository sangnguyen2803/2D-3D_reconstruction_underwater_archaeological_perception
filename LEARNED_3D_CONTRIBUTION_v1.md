# Learned 3D Contribution: Improvement and Limitations v1

## Decision

A broad learned-model improvement over the strongest classical reconstruction is **not established**. The new training and controlled evaluation provide two localized geometric gains, with adverse results retained. Keep low-contrast RootSIFT as the reconstruction reference and keep learned ranking optional. Do not replace the default with a model selected from the best test row.

- Conditional context filtering reduces median camera rotation error on block599 from 1.6123 to 1.3228 degrees (17.95%), with 32/32 registered views and improvements against ratio/random controls. This is a secondary exploratory metric; ATE does not improve there.
- Frozen SegFormer endpoint ranking on the original contrast 0.04 detector reduces block599 ATE by 2.36% against the full baseline, 5.59% against matched-budget ratio ranking and 4.38% against random ranking. All three paired unadjusted intervals exclude zero. This is one previously examined 32-image region, with 22 registered cameras, and the gain does not transfer to the stronger contrast 0.01 baseline.

## What changed

1. Conservative match-budget selection preserves every detector keypoint and ranks only mutual ratio 0.8 survivors. Every pair retains 90% or at least 128 candidates when available. Equal per-pair counts across controls isolate ranking from budget reduction.
2. Reference precision supervision uses a soft 0.5px epipolar-error target, not RANSAC labels. Original-pair, expanded MLP and expanded context models were tested separately. Expanded training uses 226 train views and 65 validation views.
3. A fresh conditional ten-input model learns within the strong contrast 0.01 survivor distribution: 455,158 matches / 388 train pairs; 221 validation pairs. Features are descriptor distance/ratio, local reliability and normalized pair coordinates. No DINO or SegFormer score enters it. Joint axis augmentation and validation-only architecture/checkpoint selection select context epoch 4.
4. A further observability model warm-starts that context checkpoint. Its train/validation target multiplies epipolar precision by min(1,(sin(reference ray angle)/sin(4deg))^2). The 4-degree scale comes from the existing mapper initialization angle, and validation retained utility selects among ten epochs. This is a reference-derived triangulation-information proxy, not measured 3D point error.
5. Optional profiles and isolated checkpoint folders preserve the existing v1 threshold path. Ten-input conditional checkpoints use standalone benchmark entry points; they are not interchangeable with the older twelve-input filters.

## Completed experiment inventory

**126 new complete sparse reconstruction runs** across six experiment families, three blocks and mapper seeds42/43/44. This count excludes the previous 54-run baseline and 9-run detector experiment. There are 96 distinct evaluated RGB images, not 126 distinct scenes or a full 1,244-view reconstruction.

| Family | New runs | Artifact |
|---|---:|---|
| Original-budget ranking | 54 | `outputs/geometry_ranking_v1` / comparison SHA256 `77d88c6cc975da233018f1415f5f42bf1ae0d8acff8922ff6ff7f959178bac12` |
| Expanded precision MLP | 18 | `outputs/geometry_ranking_expanded_v1` / comparison SHA256 `bb6b94920d3df497bedee278fdaf79d316592c1e9cd9faac7532443baf462645` |
| Expanded precision context | 9 | `outputs/geometry_context_ranking_v1` / comparison SHA256 `7a2c88d15e41fe8cc93017ac4d869b1d55383f6bc53ada7a68383d953e83dffa` |
| SegFormer / strong detector | 27 | `outputs/segformer_strong_baseline_v1` / comparison SHA256 `2f79783c1355d64e3063c6c2791ea1fc1aee9ac220f6324274865a09d8a01ad9` |
| Conditional precision | 9 | `outputs/conditional_precision_geometry_v1` / comparison SHA256 `0698ee907ece95a1bb216cdbcd5af433433f845bbdc61fe5a9670f7f7e220dd9` |
| Triangulation utility | 9 | `outputs/observability_geometry_v1` / comparison SHA256 `ca6fdd1491e537eda7e40216cd223679ba25618e3d65ea8ad0504c5019703f80` |

## Strong baseline: primary camera-position accuracy

All methods below register 32/32 cameras per block/seed. ATE is Sim3-aligned camera-center RMSE in published reference units, not verified metres. Values are three-seed means. Lower is better. Ratio90/random90 share the exact learned match budget; full RootSIFT retains the entire eligible pool.

| Block | Full RootSIFT .01 | Ratio90 | Random90 | SegFormer90 | Conditional90 | Observability90 |
|---|---:|---:|---:|---:|---:|---:|
| block273 | 0.021124 | 0.019383 | 0.019991 | 0.020529 | 0.020712 | 0.020432 |
| block599 | 0.022459 | 0.022842 | 0.022345 | 0.023515 | 0.022764 | 0.022678 |
| block1116 | 0.017402 | 0.017168 | 0.017462 | 0.020101 | 0.017643 | 0.016922 |

Use the paired ATE intervals in [SegFormer strong-baseline assessment](SEGFORMER_STRONG_BASELINE_v1.md), [conditional precision assessment](CONDITIONAL_PRECISION_GEOMETRY_v1.md) and [observability assessment](OBSERVABILITY_GEOMETRY_v1.md). Full registration alone does not establish geometric accuracy. Native reprojection and point counts use differing point populations and cannot replace paired pose evaluation.

## Secondary angular audit: all regions and controls

Camera orientations are aligned using the same center-derived Sim3 rotation. The table averages per-seed median angular errors. Intervals use 500 paired circular moving blocks of four cameras, refitting Sim3 in each draw and averaging the same draws across seeds. These are exploratory, conditional intervals without multiple-comparison adjustment, not a primary success declaration.

| Block | Control | Control degrees | Conditional degrees | Reduction | 95% delta CI degrees |
|---|---|---:|---:|---:|---|
| block273 | low_contrast_ratio | 0.5133 | 0.5734 | -11.69% | [-0.0170, 0.1348] |
| block273 | ratio90 | 0.5577 | 0.5734 | -2.81% | [-0.0689, 0.0566] |
| block273 | random90 | 0.4898 | 0.5734 | -17.05% | [-0.0155, 0.1724] |
| block599 | low_contrast_ratio | 1.6123 | 1.3228 | 17.95% | [-0.4177, -0.0645] |
| block599 | ratio90 | 1.7173 | 1.3228 | 22.97% | [-0.4850, -0.0221] |
| block599 | random90 | 1.6328 | 1.3228 | 18.99% | [-0.4127, -0.0504] |
| block1116 | low_contrast_ratio | 0.5651 | 0.5889 | -4.20% | [-0.0216, 0.0573] |
| block1116 | ratio90 | 0.5704 | 0.5889 | -3.24% | [-0.0266, 0.0495] |
| block1116 | random90 | 0.5660 | 0.5889 | -4.04% | [-0.0285, 0.0509] |

## Interpretation for interviews and CV

Keep classification AP, calibration and reconstruction claims separate. AP 0.9473 and SegFormer fold AP 0.84-0.89 remain classification results. The strongest broad coverage/ATE change remains the detector improvement. Learned geometry evidence is currently localized and exploratory.

A defensible technical statement is: **Developed reference-supervised context filtering and controlled SfM ablations; reduced median camera rotation error by 18% on one 32-view region against a strong RootSIFT baseline, while identifying limited transfer to positional accuracy and other regions.**

Do not state that learned filtering improves full-dataset 3D reconstruction or dense mesh accuracy. The pose reference is published bundle adjustment, not independently surveyed geometry. Test regions were already examined in earlier development, and many scorers were tested. A new acquisition or genuinely untouched region is required for a confirmatory claim. No new dense-MVS run or ground-truth surface evaluation was performed in this task.

## Reproduction and next evidence

Run commands from the repository root with output paths that do not exist. Executed source copies and checkpoint hashes accompany the new conditional/observability experiments. The original-budget benchmark launched before cosmetic formatting; its archived formatted source is not the exact launch bytes, a limitation recorded here.

```powershell
.venv\Scripts\python.exe scripts/train_conditional_precision.py --output outputs/conditional_reproduction_v1
.venv\Scripts\python.exe scripts/train_observability_ranker.py --source outputs/conditional_reproduction_v1 --output outputs/observability_reproduction_v1
.venv\Scripts\python.exe scripts/benchmark_conditional_precision.py --training outputs/observability_reproduction_v1 --candidate-cache outputs/conditional_precision_geometry_v1 --output outputs/observability_geometry_reproduction_v1
```

These benchmark scripts intentionally use the existing locked images, pools and control artifacts; the reproduction command is not a general new-dataset benchmark. For a substantive next accuracy claim, reserve an untouched region/acquisition, lock the primary metric and model before evaluation, and supervise held-out reprojection or triangulated 3D error. Do not continue selecting architectures from these same three test blocks.

Validation: 119 tests passed; Ruff and Black checks passed. Prior summaries and results remain preserved. No commit/push or checkpoint deletion was performed.

Documentation clarification: generic closing reproduction paragraphs in earlier individual reports refer to the historical twelve-input .04 ranker. The conditional 10-input and strong 0.01 experiments use the distinct scripts and protocols described in this report; their numeric tables remain valid.
