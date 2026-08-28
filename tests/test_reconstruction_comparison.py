import numpy as np

from underwater_vision.evaluation.reconstruction_comparison import compare_poses


def test_common_camera_population_and_similarity_invariance():
    rng = np.random.default_rng(12)
    centers = rng.normal(size=(32, 3))

    def poses(values):
        result = {}
        for i, value in enumerate(values):
            pose = np.eye(4)
            pose[:3, 3] = value
            result[f"image{i:03}"] = pose
        return result

    reference = poses(centers)
    before = poses(3 * centers + 7 + rng.normal(scale=0.1, size=centers.shape))
    after = poses(2 * centers - 4)
    after.pop("image031")
    result = compare_poses(before, after, reference, repeats=50)
    assert result["common_cameras"] == 31
    assert result["registered_before"] == 32
    assert result["registered_after"] == 31
    assert result["ate_after"] < 1e-12
    assert result["ci95_delta"][1] < 0
