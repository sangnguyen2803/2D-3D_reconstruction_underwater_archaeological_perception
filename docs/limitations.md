# Limitations and failure analysis

- Camera poses are bundle-adjustment reference estimates, not independent measured
  camera ground truth. Agreement can inherit biases from the original photogrammetry.
- The GCP STL is a reference network, not a dense surface survey. No dense accuracy
  or completeness metrics can honestly be derived from it.
- The PDF omits an explicit calibration/pose convention statement. Center-offset
  Brown calibration and camera-to-world transforms are documented interpretations.
- A GoPro in water is not a perfect central camera. A Brown model does not explicitly
  model refractive interfaces, housing geometry, or wavelength-dependent propagation.
- DINO patch descriptors are coarser than keypoints, and interpolation does not create
  subpixel geometric precision. SIFT provides localization; the neural baseline is hybrid.
- Handcrafted visibility can suppress useful smooth/low-contrast archaeological regions.
  A high inlier ratio can accompany fewer points and worse coverage.
- Geometric weak supervision can reinforce incorrect initial geometry. Soft-label
  validation loss is not independent reconstruction-error calibration.
- Temporal gaps do not ensure spatial separation in a diver's looping trajectory.
- COLMAP bundle adjustment and PatchMatch are not reliability-weighted internally.
  Matching graph selection and post-reconstruction confidence are the custom integration.
- Confidence parameters and depth tolerances need sensitivity studies. Their values
  are not calibrated probabilities and should not be used as metrology guarantees.
- Development subset experiments do not establish performance across all Mermaid views.
  Multiple sequences/seeds and a locked test protocol remain necessary.
- Optional semantic segmentation, Point Transformer, and service deployment are outside
  the implemented core; the dataset does not supply verified semantic labels.

Observed run failures and measured regressions are recorded in PROJECT_STATUS.md and
the experiment run directories. The technical report must be updated when additional
experiments supply evidence; it should never infer success from attractive screenshots.
