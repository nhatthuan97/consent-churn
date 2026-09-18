"""Shared experiment setup: one definition of how the data is split and how
silos are formed.

Every experiment in this repo starts from the same two steps -- an 80/20
stratified split with the numeric columns standardised on train statistics,
and a Dirichlet partition of the training set into K silos. Those two steps
used to be copy-pasted into `federated_methods.py`, `churn.py` and
`mlp_amplification_study.py` as byte-identical blocks. That is a hazard rather
than a convenience: the silo partition is what the central "who leaves matters
more than how many" result is measured against, so three copies means a change
to it can silently make the demo and the study disagree about what a silo is.

Both functions are deterministic in their seed; behaviour is unchanged from the
copies they replace, and `mlp_amplification_results.json` still regenerates
byte-for-byte identical.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

# best_single_baseline owns the preprocessing (load_xy / NUMERIC).
sys.path.insert(0, str(Path(__file__).resolve().parent / "01_baseline_fullscale"))


def load_split_standardize(test_size=0.2, seed=0):
    """Load the cohort, split it stratified, standardise the numeric columns.

    Returns dict(Xtr, ytr, Xte, yte, nf). Numeric columns are scaled with the
    TRAIN mean/sd only, so no test statistics leak into training.
    """
    from sklearn.model_selection import train_test_split
    from best_single_baseline import load_xy, NUMERIC

    X, y, feat = load_xy()
    num_idx = [i for i, n in enumerate(feat) if n in NUMERIC]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y,
                                          random_state=seed)
    mu = Xtr[:, num_idx].mean(0); sd = Xtr[:, num_idx].std(0) + 1e-8
    Xtr = Xtr.copy(); Xte = Xte.copy()
    Xtr[:, num_idx] = (Xtr[:, num_idx] - mu) / sd
    Xte[:, num_idx] = (Xte[:, num_idx] - mu) / sd
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, nf=X.shape[1])


def partition(y, k, alpha, seed=0):
    """Dirichlet label partition of the training indices into `k` silos.

    Drawing a separate Dirichlet proportion per class is what makes silos differ
    in class BALANCE, not just in size -- small alpha gives the positive-heavy /
    positive-light silos the whole-silo-exit experiment depends on.
    """
    rng = np.random.default_rng(seed); silos = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); pr = rng.dirichlet(alpha * np.ones(k))
        cuts = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
        for s, ch in enumerate(np.split(idx, cuts)): silos[s].extend(ch.tolist())
    return [np.sort(np.array(s, int)) for s in silos]
