# Visibility-Aware Underwater 2D–3D Reconstruction Pipeline

- Formulated dense keypoint matchability prediction from reference-pose-derived epipolar labels on the **1,244-image EPITA Mermaid dataset**, auditing **94% of matches below 1.5 px** and using spatial train/validation/test splits.
- Trained **SegFormer-B0**, outperforming frozen **DINOv2+MLP in 2 of 3 spatial folds** (paired 95% CI; mean AP **0.84–0.89**); reduced mean **ECE from 10.60% to 3.50%** with validation-only Platt scaling.
- Developed **MLP and context-attention correspondence filters**, with the MLP reaching **AP 0.947** on held-out candidates versus **0.923** for ratio-test scoring.
- Optimized **RootSIFT** extraction, reducing reference-aligned pose error by **20.2%** and doubling sparse 3D points on a **32-view benchmark across three seeds**.

## LaTeX

```latex
\item Formulated dense keypoint matchability prediction from reference-pose-derived epipolar labels on the \textbf{1,244-image EPITA Mermaid dataset}, auditing \textbf{94\% of matches below 1.5 px} and using spatial train/validation/test splits.

\item Trained \textbf{SegFormer-B0}, outperforming frozen \textbf{DINOv2+MLP in 2 of 3 spatial folds} (paired 95\% CI; mean AP \textbf{0.84--0.89}); reduced mean \textbf{ECE from 10.60\% to 3.50\%} with validation-only Platt scaling.

\item Developed \textbf{MLP and context-attention correspondence filters}, with the MLP reaching \textbf{AP 0.947} on held-out candidates versus \textbf{0.923} for ratio-test scoring.

\item Optimized \textbf{RootSIFT} extraction, reducing reference-aligned pose error by \textbf{20.2\%} and doubling sparse 3D points on a \textbf{32-view benchmark across three seeds}.
```

## Evidence notes

The 94% figure is from the reference-geometry audit, not the fraction of all raw candidate matches that are correct. The AP range gives three-seed fold means, not every individual run. ECE is expected calibration error, not accuracy. The AP 0.947 result belongs to the MLP, not LightGlue or context-attention. The RootSIFT result is a classical detector improvement on one of three evaluated 32-image blocks; it is not a SegFormer benefit or a completed full-dataset reconstruction. Pose errors use published bundle-adjusted reference poses, not independent surveyed ground truth.

Detailed results and limitations: [Project summary v4](PROJECT_SUMMARY_v4.md).
