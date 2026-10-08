import numpy as np

from underwater_vision.retrieval.pairs import retrieve_pairs


def test_retrieval_never_contains_self_pairs_or_duplicates():
    descriptors = np.random.default_rng(2).normal(size=(20, 8))
    pairs, neighbors = retrieve_pairs(descriptors, 3)
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == len(pairs)
    assert all(i not in row for i, row in enumerate(neighbors))
