import numpy as np
import torch

from underwater_vision.evaluation.uncertainty import binary_metrics
from underwater_vision.matching.learned_filter import MatchFilter


def test_context_filter_is_permutation_equivariant():
    torch.manual_seed(3)
    model = MatchFilter("context").eval()
    x = torch.randn(30, 12)
    order = torch.randperm(30)
    with torch.no_grad():
        np.testing.assert_allclose(model(x)[order], model(x[order]), atol=1e-6)
    assert model(torch.empty(0, 12)).shape == (0,)


def test_binary_metrics_exclude_ambiguous_labels():
    result = binary_metrics([0, 1, -1, 1], [0.1, 0.9, 0.5, 0.8])
    assert result["samples"] == 3
    assert result["roc_auc"] == 1
    assert result["average_precision"] == 1
