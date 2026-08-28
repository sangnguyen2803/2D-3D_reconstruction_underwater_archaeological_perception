# Project Specification

## Visibility-Aware 2D–3D Reconstruction for Underwater Archaeological Perception

## 1. Project Objective

Build an end-to-end Computer Vision research/engineering project for underwater scene reconstruction.

The central research question is:

> **How can underwater image visibility and feature reliability be explicitly modeled to improve multi-view 3D reconstruction under underwater image degradation?**

The system should take underwater RGB images acquired from a moving camera and produce:

1. Visibility-aware 2D feature representations.
2. Robust cross-view feature correspondences.
3. Camera poses / trajectory estimation.
4. Sparse 3D reconstruction.
5. Dense 3D reconstruction.
6. A colored 3D point cloud / mesh.
7. A per-point or per-region reconstruction confidence map.
8. Optional semantic/object-level understanding of the underwater archaeological scene.
9. Quantitative comparison against standard non-visibility-aware reconstruction baselines.

The project should emphasize:

* 2D Computer Vision
* foundation-model visual features
* multi-view geometry
* Structure-from-Motion
* Multi-View Stereo
* 3D point clouds
* confidence estimation
* cross-modal 2D/3D reasoning
* reproducible engineering
* experiment tracking
* deployment/demo

The project must NOT simply be:

> "Run COLMAP on an underwater dataset."

The main contribution should be a **visibility/reliability-aware reconstruction pipeline**.

---

# 2. Dataset

## Primary Dataset: Mermaid Underwater Dataset

Use the official Mermaid Underwater Dataset from EPITA/LRE.

Official dataset information:

https://www.lre.epita.fr/productions/mermaid/

The dataset contains:

* 1,250 high-resolution RGB images
* image resolution: 3840 × 2800
* approximately 15 GB
* underwater scene around 18.7–19 m depth
* approximately 150 m² surveyed area
* natural lighting
* one camera
* camera calibration
* camera trajectories
* camera poses
* ground-control information
* 5 ground control points
* archaeological Roman tile fragments
* a mermaid statue
* rocky areas
* marine life

The dataset is explicitly intended for underwater photogrammetry, 3D reconstruction, SfM, MVS, visual SLAM, feature matching, 6-DoF pose estimation, relocalization and related tasks.

Use the official source as the authoritative description of the dataset.

The dataset is suitable because it provides both RGB imagery and geometric/camera metadata rather than only image-level labels.

---

# 3. Core Research Problem

Underwater imagery introduces degradation that can make standard multi-view reconstruction unreliable.

Relevant effects include:

* wavelength-dependent color attenuation
* blue/green color shift
* backscatter
* reduced contrast
* haze-like veiling
* non-uniform illumination
* low-texture regions
* marine vegetation
* suspended particles
* specular/reflection effects
* viewpoint-dependent visibility

The project should therefore treat visual correspondence as a **reliability estimation problem**.

Instead of:

```
feature → match → reconstruction
```

use:

```
feature
   +
visibility/reliability
   ↓
weighted matching
   ↓
weighted geometric estimation
   ↓
reconstruction
   ↓
confidence-aware 3D map
```

---

# 4. Main Hypothesis

The system should test the following hypothesis:

> **Not all image features should contribute equally to underwater geometric reconstruction. Explicitly estimating feature reliability from local visual evidence can improve correspondence quality, camera estimation, and final 3D reconstruction.**

The project should therefore compare:

### Baseline

```
RGB
 ↓
classical/SOTA feature matching
 ↓
SfM
 ↓
MVS
```

### Proposed

```
RGB
 ↓
visual features
 +
visibility/reliability estimation
 ↓
reliability-weighted correspondence
 ↓
reliability-aware geometric estimation
 ↓
SfM
 ↓
MVS
 ↓
confidence-aware point cloud
```

The experiment should demonstrate whether the additional reliability modeling actually helps.

---

# 5. High-Level Architecture

Implement the following conceptual pipeline:

```
┌─────────────────────────────┐
│   Underwater RGB Images     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Image preprocessing          │
│ - resize                     │
│ - color normalization        │
│ - optional enhancement       │
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌─────────────┐   ┌────────────────┐
│ DINOv2      │   │ Visibility      │
│ features    │   │ estimation      │
└──────┬──────┘   └───────┬────────┘
       │                  │
       └──────────┬───────┘
                  ▼
      ┌───────────────────────┐
      │ Reliability-aware     │
      │ feature matching      │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ Geometric verification│
      │ RANSAC / epipolar     │
      │ constraints            │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ Structure-from-Motion │
      │ camera poses + sparse │
      │ point cloud            │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ Multi-View Stereo     │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ Dense 3D point cloud  │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ 3D confidence map     │
      └───────────┬───────────┘
                  ▼
      ┌───────────────────────┐
      │ 3D visualization      │
      └───────────────────────┘
```

---

# 6. Foundation Model

## DINOv2

Use DINOv2 as the primary visual feature extractor.

Official implementation:

https://github.com/facebookresearch/dinov2

DINOv2 provides pretrained ViT backbones and dense visual representations.

Recommended starting model:

```
DINOv2 ViT-B/14
```

Do not start with ViT-L or ViT-g.

Reason:

* ViT-B/14 is substantially cheaper.
* The project is about the reconstruction pipeline, not backbone scaling.
* ViT-B is sufficient for an initial experiment.
* Larger models can be tested later as an ablation.

Keep DINOv2 frozen initially.

Extract patch-level features rather than only a global CLS embedding.

For example:

```
RGB image
    ↓
DINOv2 ViT-B/14
    ↓
patch feature grid
    ↓
H' × W' × C
```

The feature map should later be associated with image coordinates.

---

# 7. Feature Matching

Do not assume DINOv2 alone is a complete feature matcher.

The system should support two feature-matching modes.

## Baseline matcher

Use a standard geometric feature pipeline such as:

* SIFT
* ORB if necessary
* LightGlue/SuperPoint if available

The purpose is to establish a strong practical baseline.

## Proposed matcher

Combine:

```
local visual descriptor
+
DINOv2 semantic/local feature
+
visibility score
```

Conceptually:

```
descriptor_i
descriptor_j
     ↓
similarity
     ↓
visibility_i
visibility_j
     ↓
weighted similarity
```

For example:

```
S_ij = cosine(F_i, F_j)
```

and define:

```
W_ij = R_i × R_j
```

then:

```
S'_ij = W_ij × S_ij
```

where R_i and R_j represent local feature reliability.

Do NOT claim this formula itself is novel.

The novelty is the complete reliability-aware reconstruction formulation and its empirical evaluation.

---

# 8. Visibility / Reliability Estimation

This is the main custom module.

The system should estimate a reliability score:

```
R(x,y) ∈ [0,1]
```

for each image region or feature.

The first implementation should NOT require a dedicated manually annotated visibility dataset.

Instead, build a weak/self-supervised reliability estimator from measurable image properties.

Candidate signals:

1. Local contrast
2. Gradient magnitude
3. Laplacian sharpness
4. local entropy
5. saturation
6. brightness
7. color-channel imbalance
8. local texture strength
9. feature repeatability
10. cross-view consistency

Construct:

```
R = f(
    contrast,
    sharpness,
    texture,
    color statistics,
    feature consistency
)
```

Start with a deterministic reliability score.

Then implement a learned reliability head as a second version.

---

# 9. Reliability Features

For every local image patch/keypoint calculate:

### 9.1 Sharpness

Use Laplacian variance or gradient statistics.

Low sharpness:

```
low reliability
```

High sharpness:

```
higher reliability
```

### 9.2 Local contrast

Compute local standard deviation / contrast.

Low-contrast underwater regions should generally receive lower reliability.

### 9.3 Local entropy

Texture-rich regions tend to provide more useful geometric evidence than uniform regions.

### 9.4 Color statistics

Calculate:

* RGB means
* RGB standard deviation
* channel ratios
* saturation
* brightness

Do not assume that a simple color heuristic is always physically meaningful. Treat these as auxiliary features.

### 9.5 Feature repeatability

This is more important.

If a local feature repeatedly matches consistently across nearby frames/views, its reliability should increase.

If it produces unstable matches, its reliability should decrease.

This produces a more geometry-aware reliability signal.

---

# 10. Learned Reliability Head

After implementing the deterministic baseline, implement:

```
DINOv2 patch feature
         ↓
     MLP head
         ↓
   reliability score
```

Architecture:

```
C
↓
Linear(C, C/2)
↓
GELU
↓
Linear(C/2, 1)
↓
Sigmoid
```

The head should predict:

```
R ∈ [0,1]
```

Training signal can initially come from geometric consistency.

For example:

A feature that repeatedly produces:

* low reprojection error
* consistent epipolar geometry
* stable cross-view correspondence

is treated as a positive reliability example.

A feature producing:

* high reprojection error
* inconsistent correspondence
* unstable triangulation

is treated as low reliability.

This creates a weak/self-supervised learning loop.

---

# 11. Cross-View Consistency

This is a key component.

For image pair:

```
I_i, I_j
```

estimate correspondences.

For each candidate correspondence:

```
p_i ↔ p_j
```

calculate:

* descriptor similarity
* epipolar error
* geometric consistency
* local visibility/reliability
* reprojection error

Store:

```
correspondence confidence
```

Then use the confidence during geometric estimation.

---

# 12. Geometric Verification

Never trust neural feature matching alone.

Apply classical geometry.

Use:

* fundamental matrix
* essential matrix when intrinsics are known
* RANSAC
* epipolar distance
* triangulation
* reprojection error

A candidate match should survive:

```
descriptor test
      +
visibility test
      +
geometric verification
```

This is important because the project should demonstrate understanding of real Computer Vision systems rather than only pretrained neural networks.

---

# 13. Structure-from-Motion

Use COLMAP as the primary reconstruction backend.

Do not reimplement a complete SfM system.

COLMAP should handle:

* feature management
* matching
* geometric verification
* camera pose estimation
* triangulation
* bundle adjustment

Your custom contribution should be upstream of the reconstruction backend.

The system should produce:

```
camera poses
sparse point cloud
reprojection errors
track information
```

The pipeline should preserve these outputs for analysis.

---

# 14. Custom Integration With COLMAP

The agent should create an adapter layer:

```
project/
    reconstruction/
        colmap/
            database.py
            features.py
            matches.py
            mapper.py
            exporter.py
```

The adapter should allow:

```
standard features
vs
reliability-aware features/matches
```

to be inserted into the same COLMAP reconstruction pipeline.

This makes ablation experiments straightforward.

---

# 15. Multi-View Stereo

After SfM:

```
sparse reconstruction
      ↓
     MVS
      ↓
dense reconstruction
```

Use COLMAP's dense reconstruction initially.

Potential later alternatives:

* PatchMatch stereo
* OpenMVS
* modern neural MVS

Do NOT implement neural MVS until the classical pipeline works.

The project must first establish:

```
RGB → reliable matching → SfM → MVS → point cloud
```

---

# 16. 3D Confidence

Every reconstructed 3D point should ideally have a confidence estimate.

Possible inputs:

* number of supporting views
* triangulation angle
* reprojection error
* source-image visibility
* feature reliability
* MVS consistency

Example:

```
C_point =
    weighted combination of
    view support
    reprojection quality
    triangulation quality
    feature reliability
```

Normalize to:

```
C_point ∈ [0,1]
```

Visualization:

```
high confidence → strong point
low confidence  → weak/translucent point
```

Do not hard-code arbitrary thresholds without validating them.

---

# 17. Optional 3D Feature Encoder

Only after the reconstruction pipeline works, add a 3D deep-learning component.

Recommended:

* Point Transformer
* PointNet++
* sparse 3D CNN

Preferred:

```
Point Transformer
```

because it is more representative of modern 3D perception than a basic PointNet implementation.

Input:

```
XYZ
RGB
confidence
```

For each point:

```
[x, y, z, r, g, b, confidence]
```

Possible tasks:

1. confidence-aware point classification
2. scene/object segmentation
3. archaeological object identification
4. reconstruction quality prediction

Do not force a classification task if the dataset does not provide reliable semantic labels.

---

# 18. Archaeological Object Understanding

The Mermaid scene contains:

* mermaid statue
* Roman tile fragments
* seabed
* rock
* marine life

The project should NOT turn into a generic object detector.

Instead, optionally demonstrate:

```
reconstructed 3D scene
      ↓
region/object segmentation
      ↓
archaeological region isolation
```

If annotations are insufficient, use manual evaluation or weak labels only for visualization.

Do not fabricate ground truth.

---

# 19. Experiments

The project must include controlled experiments.

## Experiment A — Classical baseline

```
SIFT
  ↓
matching
  ↓
COLMAP
  ↓
SfM/MVS
```

## Experiment B — Neural feature baseline

```
DINOv2
  ↓
matching
  ↓
COLMAP
```

## Experiment C — Visibility-aware

```
DINOv2
  +
deterministic reliability
  ↓
weighted matching
  ↓
COLMAP
```

## Experiment D — Learned reliability

```
DINOv2
  +
learned reliability head
  ↓
weighted matching
  ↓
COLMAP
```

These experiments are mandatory.

---

# 20. Evaluation Metrics

Evaluate both 2D matching and 3D reconstruction.

## 20.1 Feature matching

Report:

* number of matches
* inlier ratio
* geometric verification success
* median epipolar error

## 20.2 Camera estimation

Use available ground-truth camera poses.

Report:

* rotation error
* translation error
* Absolute Trajectory Error where appropriate
* Relative Pose Error

## 20.3 Sparse reconstruction

Report:

* number of registered images
* number of reconstructed points
* reprojection error
* track length

## 20.4 Dense reconstruction

Where ground-truth geometry is available, evaluate:

* point-to-point distance
* point-to-plane distance
* Chamfer distance
* completeness
* accuracy
* F-score at selected distance thresholds

Do not report metrics that cannot be computed reliably from the available ground truth.

---

# 21. Ablation Study

The final report must include:

### A. No reliability

```
DINOv2 → matching → SfM
```

### B. Handcrafted reliability

```
DINOv2 + image-quality features
```

### C. Feature consistency

```
DINOv2 + cross-view reliability
```

### D. Learned reliability

```
DINOv2 + learned reliability
```

### E. Different backbones

Compare:

```
SIFT
DINOv2-S
DINOv2-B
```

Only run larger models if computationally practical.

---

# 22. Critical Evaluation

Do not assume the proposed method will outperform every baseline.

If reliability weighting hurts reconstruction, investigate why.

Possible failure modes:

* visibility score suppresses useful low-contrast features
* underwater color statistics are misleading
* DINOv2 features are semantically strong but geometrically weak
* weighting removes necessary features
* repeated textures create false correspondences
* marine vegetation produces unstable geometry
* insufficient baseline between views
* low-texture seabed

The final report must explicitly document failures.

This is important for research credibility.

---

# 23. Data Split

Avoid random image-level train/test splitting for reconstruction evaluation.

Images are temporally/spatially correlated.

Use sequence/trajectory-aware splits.

Recommended:

### Reconstruction split

```
earlier/selected trajectory
    →
reconstruction

held-out trajectory/images
    →
evaluation
```

Where possible, evaluate on spatially separated views rather than random neighboring frames.

This prevents leakage.

---

# 24. Image Resolution Strategy

Do NOT run the entire project at:

```
3840 × 2800
```

during experimentation.

Use a staged strategy.

### Development

```
960 × 700
or
1280 × 933
```

### Final reconstruction

Use higher resolution selectively.

For example:

```
Stage 1:
low-resolution feature extraction

Stage 2:
high-resolution local matching

Stage 3:
full-resolution geometric refinement
```

This reduces GPU memory and makes iteration practical.

---

# 25. Recommended Software Stack

## Core

* Python 3.11
* PyTorch
* torchvision
* NumPy
* SciPy
* OpenCV
* Pillow

## Deep Learning

* PyTorch
* DINOv2
* xFormers if needed

DINOv2 official repository:

https://github.com/facebookresearch/dinov2

## Feature Matching

Prefer:

* LightGlue
* SuperPoint

Use SIFT as classical baseline.

## Geometry

* OpenCV
* SciPy
* pycolmap where appropriate

## Reconstruction

* COLMAP

## Point Clouds

* Open3D
* PyTorch3D only if needed

Open3D should be the default visualization/processing library.

## Experiment tracking

* MLflow

Track:

* experiment configuration
* metrics
* model checkpoint
* reconstruction statistics
* qualitative outputs

## Configuration

Use:

* Hydra
* YAML configuration

## Testing

* pytest

## Code quality

* ruff
* black
* mypy where useful

## API/demo

Optional:

* FastAPI
* Uvicorn

## Visualization

Use:

* Open3D
* matplotlib
* Plotly only if useful

Do not make a web frontend a priority.

---

# 26. Repository Architecture

Use a professional structure:

```
underwater-vision/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── data/
│   ├── model/
│   ├── matching/
│   ├── reconstruction/
│   └── experiments/
│
├── src/
│   └── underwater_vision/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── visibility/
│       ├── matching/
│       ├── geometry/
│       ├── reconstruction/
│       ├── pointcloud/
│       ├── evaluation/
│       ├── visualization/
│       └── utils/
│
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── estimate_visibility.py
│   ├── match_features.py
│   ├── run_sfm.py
│   ├── run_mvs.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_visibility_analysis.ipynb
│   ├── 03_feature_matching.ipynb
│   ├── 04_reconstruction_analysis.ipynb
│   └── 05_results.ipynb
│
├── tests/
│
├── outputs/
│   ├── features/
│   ├── matches/
│   ├── sparse/
│   ├── dense/
│   └── figures/
│
└── docs/
    ├── methodology.md
    ├── experiments.md
    └── limitations.md
```

---

# 27. Engineering Requirements

The agent must implement:

### Reproducibility

Every experiment should be runnable with:

```
python scripts/run_experiment.py \
    experiment=visibility_aware
```

Configuration should control:

* dataset path
* image resolution
* feature extractor
* matcher
* visibility method
* RANSAC thresholds
* COLMAP settings
* random seed

---

# 28. Caching

Feature extraction is expensive.

Cache:

```
DINOv2 features
```

to disk.

Recommended structure:

```
cache/
    dinov2/
        image_000001.pt
        image_000002.pt
```

Similarly cache:

```
keypoints
descriptors
visibility maps
pairwise matches
```

Never recompute expensive features unnecessarily.

---

# 29. GPU Strategy

The project must work on a single consumer GPU.

Recommended initial target:

```
NVIDIA GPU
8–16 GB VRAM
```

Use:

* mixed precision
* batch processing
* feature caching
* resized images
* CPU offloading where appropriate

Do not require multi-GPU training.

Do not train DINOv2 from scratch.

Do not train a large 3D foundation model from scratch.

---

# 30. Development Phases

## Phase 0 — Dataset

Goal:

Load Mermaid data and understand:

* images
* calibration
* camera poses
* trajectories
* GCP information

Deliverable:

```
dataset_loader.py
```

---

## Phase 1 — Classical reconstruction

Build:

```
images
  ↓
SIFT
  ↓
COLMAP
  ↓
sparse reconstruction
  ↓
dense reconstruction
```

Deliverable:

A valid underwater 3D reconstruction.

This must work before any deep-learning contribution.

---

## Phase 2 — DINOv2

Add:

```
RGB
 ↓
DINOv2
 ↓
patch features
```

Analyze:

* feature similarity
* feature repeatability
* correspondence quality

---

## Phase 3 — Reliability estimation

Implement deterministic reliability.

Produce:

```
RGB
 ↓
reliability heatmap
```

Visualize reliability over underwater scenes.

---

## Phase 4 — Reliability-aware matching

Implement:

```
DINOv2
   +
reliability
   ↓
weighted matching
```

Compare against baseline.

---

## Phase 5 — Geometry integration

Integrate weighted matches with:

* RANSAC
* essential/fundamental matrix estimation
* COLMAP

Measure whether:

* image registration improves
* inlier ratio improves
* reprojection error decreases
* reconstruction completeness improves

---

## Phase 6 — Learned reliability

Replace or augment the handcrafted reliability model with a trainable MLP.

Train using geometric consistency as weak supervision.

---

## Phase 7 — 3D confidence

Compute per-point confidence.

Produce:

```
confidence-aware point cloud
```

---

## Phase 8 — Final evaluation

Run complete ablations.

Produce:

* quantitative tables
* reconstruction comparisons
* confidence maps
* trajectory plots
* 3D visualizations

---

# 31. Final Demo

The final demo should show:

### Input

Several underwater images.

### Step 1

Visibility map:

```
RGB → reliability heatmap
```

### Step 2

Feature matching:

```
image A ↔ image B
```

with:

* raw matches
* filtered matches
* reliability-weighted matches

### Step 3

Camera trajectory:

```
reconstructed camera path
```

### Step 4

Sparse reconstruction:

```
sparse point cloud
```

### Step 5

Dense reconstruction:

```
colored point cloud
```

### Step 6

Confidence:

```
reconstruction confidence
```

### Step 7

Comparison:

```
baseline vs proposed
```

The most important visualization is:

```
Baseline reconstruction
         VS
Reliability-aware reconstruction
```

---

# 32. Research Contribution

The project should claim the following contribution conservatively:

> A visibility-aware 2D–3D reconstruction pipeline for underwater imagery that estimates local feature reliability and incorporates it into multi-view correspondence and geometric reconstruction.

Do NOT claim:

> "A novel SOTA underwater reconstruction model"

unless experiments genuinely demonstrate that.

The novelty should be positioned as:

1. reliability-aware feature processing,
2. integration of learned visual features with geometric constraints,
3. confidence propagation from 2D observations into 3D reconstruction,
4. systematic evaluation under underwater degradation.

---

# 33. What NOT to Do

Do NOT:

* train DINOv2 from scratch
* build a giant Transformer unnecessarily
* add an LLM
* add a VLM just because it is trendy
* use SAM2 without a clear purpose
* use Point Transformer merely for CV keyword coverage
* create a fake segmentation task
* fabricate labels
* claim SOTA without comparison
* evaluate using only qualitative screenshots
* randomly split adjacent frames
* run everything at 4K from the beginning
* build a complicated frontend before the reconstruction works

The project should prioritize scientific/engineering depth over model count.

---

# 34. Optional Advanced Extension

If the core pipeline works, implement:

## Learned uncertainty-aware matching

Instead of:

```
binary match / no match
```

predict:

```
P(match is geometrically reliable)
```

Then use this probability inside:

* weighted RANSAC
* bundle adjustment weighting
* triangulation filtering
* MVS confidence

This creates a stronger research direction:

> **Uncertainty-aware multi-view geometry for underwater perception.**

---

# 35. Optional Deployment

Build a lightweight FastAPI service:

```
POST /reconstruct
```

Input:

* image sequence
* optional camera metadata

Output:

* camera trajectory
* sparse point cloud path
* dense point cloud path
* confidence map
* metrics

Do not attempt real-time reconstruction.

The deployment goal is reproducible batch inference, not production robotics.

---

# 36. Final CV-Level Project Description

Potential final project title:

> **Visibility-Aware 2D–3D Reconstruction for Underwater Archaeological Perception**

Potential CV bullets:

> • Developed a visibility-aware multi-view reconstruction pipeline combining DINOv2 visual features with geometric consistency to improve feature matching and 3D reconstruction in degraded underwater imagery.

> • Propagated image-level feature reliability into SfM/MVS, producing confidence-aware 3D point clouds and evaluating camera pose, reprojection error, reconstruction completeness, and geometric accuracy.

Do not claim numerical improvements until they have actually been measured.

---

# 37. Definition of Done

The project is considered complete only when all of the following exist:

* [ ] Mermaid dataset loader
* [ ] reproducible preprocessing pipeline
* [ ] classical SfM/MVS baseline
* [ ] DINOv2 feature extraction
* [ ] feature matching baseline
* [ ] visibility/reliability estimator
* [ ] reliability-aware matching
* [ ] COLMAP integration
* [ ] sparse reconstruction
* [ ] dense reconstruction
* [ ] 3D confidence estimation
* [ ] quantitative evaluation
* [ ] ablation study
* [ ] qualitative visualization
* [ ] experiment configuration
* [ ] cached features
* [ ] unit tests for core modules
* [ ] reproducible README
* [ ] technical report
* [ ] final demo
* [ ] limitations/failure analysis

The project should be considered successful even if the proposed method does not beat every baseline. The critical requirement is to establish a clear hypothesis, implement a rigorous baseline, perform controlled experiments, and explain the observed behavior.

# 38. Priority Order

The agent must implement in this order:

1. Dataset loading
2. Classical reconstruction
3. Evaluation against provided camera/geometric information
4. DINOv2 feature extraction
5. Feature matching
6. Reliability estimation
7. Reliability-aware matching
8. SfM integration
9. Dense reconstruction
10. 3D confidence
11. Ablations
12. Visualization
13. Optional learned reliability
14. Optional Point Transformer
15. Optional FastAPI demo

Never start with the advanced components before proving that the baseline reconstruction works.

# 39. Overall Design Philosophy

The project should look like an actual Computer Vision research/engineering project rather than a collection of pretrained models.

The central idea is:

> **Underwater perception is not only a feature-extraction problem; it is a problem of deciding which visual evidence can be trusted for geometry.**

The final system should therefore connect:

```
2D visual representation
        ↓
visibility/reliability
        ↓
feature correspondence
        ↓
multi-view geometry
        ↓
3D reconstruction
        ↓
3D confidence
```

This connection between 2D feature reliability and downstream 3D geometry should be the project's central technical identity.
