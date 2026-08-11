import numpy as np
import torch

from underwater_vision.visibility.learned import ReliabilityHead, predict


def test_learned_head_roundtrip_and_bounds(tmp_path):
    torch.manual_seed(1)
    head = ReliabilityHead(12)
    checkpoint = tmp_path / "head.pt"
    torch.save({"channels": 12, "state_dict": head.state_dict()}, checkpoint)
    features = np.random.default_rng(4).normal(size=(8, 12)).astype(np.float32)
    scores = predict(features, checkpoint)
    assert scores.shape == (8,)
    assert np.all((0 <= scores) & (scores <= 1))
    np.testing.assert_allclose(
        scores, head.reliability(torch.from_numpy(features)).detach().numpy()
    )
