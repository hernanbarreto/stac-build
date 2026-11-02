"""segmentation.pipeline._instances_share_points — the one-pass question that lets the projection skip
its pairwise index-overlap merge (station 2026-11-02: ~25 min over 1,183 instances whose index sets are
disjoint by construction): True exactly when some point is in two instances, the answer of the pairs."""
import itertools

import numpy as np

from segmentation import pipeline as P


def _pairwise(instances):
    return any(set(a["globalIndices"]) & set(b["globalIndices"])
               for a, b in itertools.combinations(instances, 2))


def test_the_one_pass_answer_is_the_pairs_answer():
    rng = np.random.default_rng(5)
    for trial in range(200):
        n_pts = int(rng.integers(5, 60))
        k = int(rng.integers(0, 6))
        insts = []
        for _ in range(k):
            m = int(rng.integers(0, n_pts))
            idx = rng.choice(n_pts, m, replace=False).tolist()
            if m and rng.random() < 0.2:
                idx.append(idx[0])                                 # a duplicate inside one instance
            insts.append({"globalIndices": idx})
        if trial % 2 == 0 and k >= 2:                              # disjoint by construction
            perm = rng.permutation(n_pts)
            cuts = np.sort(rng.choice(np.arange(1, n_pts), k - 1, replace=False))
            insts = [{"globalIndices": part.tolist()} for part in np.split(perm, cuts)]
        assert P._instances_share_points(insts, n_pts) == _pairwise(insts)
