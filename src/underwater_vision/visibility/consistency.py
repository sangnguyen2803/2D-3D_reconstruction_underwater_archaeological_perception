import numpy as np


def update_repeatability(features, results, prior=2.0):
    """Beta posterior mean from candidate matches; unobserved points keep 0.5.

    Pair-level inlier labels are weak labels; initial geometry can be wrong.
    The second pass does not use reference evaluation poses.
    """
    counts = {n: np.zeros(len(f["xy"])) for n, f in features.items()}
    success = {n: np.zeros(len(f["xy"])) for n, f in features.items()}
    for result in results:
        for side, name in enumerate([result["a"], result["b"]]):
            ids = result["pairs"][:, side]
            np.add.at(counts[name], ids, 1)
            np.add.at(success[name], ids, result["inliers"].astype(float))
    for name, feature in features.items():
        feature["repeatability"] = ((success[name] + prior / 2) / (counts[name] + prior)).astype(
            np.float32
        )
        feature["reliability"] *= feature["repeatability"]
