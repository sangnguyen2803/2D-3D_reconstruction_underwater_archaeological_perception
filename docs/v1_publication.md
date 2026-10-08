# Reference-supervised reconstruction publication

This increment adds reference-camera geometry, spatial partitioning, learned twelve-input match filters, frozen DINO global retrieval and held-out point-error prediction. The original reconstruction remains the default. Reference epipolar compatibility and published BA cameras are proxies rather than independently surveyed truth.

Install `pip install -e ".[neural,research,matchers]"`. Run the reference audit before training. `scripts/run_research_suite.py --output outputs/reproduction_v1` orchestrates a new reproduction root and rejects an existing directory. It requires the Mermaid images/reference poses, cached pretrained models or downloads, and the configured COLMAP executable for dense stages. `scripts/build_research_report.py` generates comparison documentation from its expected historical output directories; results/weights remain local.

The historical development measurements are context match AP 0.945910, classical sparse support 1,790 points versus context 1,890, and higher reprojection residual after filtering. DINO retrieval under the co-frustum proxy did not beat the sequential control. Learned point confidence AUSE 0.491018 did not beat heuristic 0.419388, and reported intervals were undercalibrated. These findings do not establish full-scene geometric improvement. Reproduction and publication checks do not rerun or extend the historical benchmark.

The tests check reference geometry, ambiguity exclusion, spatial separation, unique retrieval pairs, contextual permutation equivariance, held-out observation isolation, tie handling and unique match retention. Downloaded data, checkpoints, NPZ/PLY artifacts, caches and environments are excluded from Git.
