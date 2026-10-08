import numpy as np

from underwater_vision.research_pipeline import unique_matches


def test_probability_filter_retains_unique_highest_scoring_pairs():
    ids = np.array([[0, 1], [1, 1], [2, 2]])
    accepted = unique_matches(ids, np.array([0.8, 0.9, 0.2]), 0.5)
    np.testing.assert_array_equal(accepted, [1])
