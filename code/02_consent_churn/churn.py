"""Consent-churn schedules for the federated trainer.

A *schedule* is a callable  r -> [K active-index arrays]  giving, for training
round r, the patients in each silo whose consent is currently active. Passed to
federated_train(..., round_silos=schedule).

Three physically distinct regimes (the thesis separates what the literature
conflates), plus a matched-random control for the key "who vs how many" test:

  transient   -- each round a fraction leaves, but rejoins later (contributions
                 re-enter the population).                Hypothesis: near-harmless.
  permanent   -- consent withdrawn for good, ramping to `rate` by the last round.
                 Population shrinks; random withdrawal => no distribution shift.
  whole_silo  -- an entire institution exits (a whole slice of the distribution
                 disappears at once).                     Hypothesis: dominates.
  matched_random -- remove the SAME NUMBER of patients as a whole-silo exit, but
                 drawn uniformly across all silos. The control that isolates
                 "who leaves" (identity/distribution) from "how many leave" (count).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_baseline_fullscale"))
from federated_methods import federated_train  # noqa: E402

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def no_churn(silos):
    return lambda r: silos


def transient_schedule(silos, rate, seed=0):
    """Each round independently deactivate ~`rate` of every silo; they can return."""
    def sched(r):
        rng = np.random.default_rng(seed * 100003 + r)
        return [s[rng.random(len(s)) >= rate] for s in silos]
    return sched


def permanent_schedule(silos, rate, rounds, seed=0, ramp=True):
    """Cumulative, irreversible withdrawal ramping to `rate` by the final round.
    Withdrawal order is random within each silo => no distribution shift (pure
    data-shrinkage effect)."""
    rng = np.random.default_rng(seed * 7919 + 1)
    orders = [rng.permutation(len(s)) for s in silos]

    def frac(r):
        return rate * ((r + 1) / rounds) if ramp else rate

    def sched(r):
        out = []
        for s, order in zip(silos, orders):
            n_out = int(round(frac(r) * len(s)))
            mask = np.ones(len(s), bool)
            mask[order[:n_out]] = False
            out.append(s[mask])
        return out
    return sched


def biased_permanent_schedule(silos, y, rate, rounds, seed=0, bias=8.0, ramp=True):
    """Permanent withdrawal where positive-class patients leave preferentially
    (bias x more likely to withdraw first). This SHIFTS the class distribution
    of the training population -- the mechanism by which churn actually costs
    utility, as opposed to random withdrawal which only shrinks the data."""
    rng = np.random.default_rng(seed * 6271 + 5)
    orders = []
    for s in silos:
        ys = y[s]
        w = np.where(ys == 1, bias, 1.0)
        # weighted order without replacement via the Gumbel-top-k / exponential trick
        key = rng.exponential(1.0, len(s)) / w        # smaller key = withdraw earlier
        orders.append(np.argsort(key))

    def frac(r):
        return rate * ((r + 1) / rounds) if ramp else rate

    def sched(r):
        out = []
        for s, order in zip(silos, orders):
            n_out = int(round(frac(r) * len(s)))
            mask = np.ones(len(s), bool)
            mask[order[:n_out]] = False
            out.append(s[mask])
        return out
    return sched


def whole_silo_schedule(silos, depart_ids, depart_round=0):
    """Named silos hold no active patients from `depart_round` onward."""
    depart_ids = set(depart_ids)
    empty = np.array([], int)

    def sched(r):
        return [empty if (r >= depart_round and i in depart_ids) else s
                for i, s in enumerate(silos)]
    return sched


def matched_random_schedule(silos, n_remove, depart_round=0, seed=0):
    """Remove exactly `n_remove` patients uniformly across ALL silos from
    `depart_round` on. Count-matched control for a whole-silo departure."""
    rng = np.random.default_rng(seed * 104729 + 3)
    pool = [(i, j) for i, s in enumerate(silos) for j in range(len(s))]
    n_remove = min(n_remove, len(pool))
    chosen = rng.choice(len(pool), size=n_remove, replace=False)
    rm = {i: [] for i in range(len(silos))}
    for k in chosen:
        i, j = pool[k]; rm[i].append(j)

    def sched(r):
        if r < depart_round:
            return silos
        out = []
        for i, s in enumerate(silos):
            mask = np.ones(len(s), bool)
            if rm[i]:
                mask[rm[i]] = False
            out.append(s[mask])
        return out
    return sched


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def evaluate_run(Xtr, ytr, Xte, yte, silos, nf, schedule, *,
                 method="fedavg", seed=0, rounds=40, model="logreg", hidden=32,
                 lr=0.1):
    """Train under a churn schedule; return metrics on the global test set."""
    p = federated_train(Xtr, ytr, silos, Xte, method=method, nf=nf, seed=seed,
                        rounds=rounds, round_silos=schedule, balanced=True,
                        model=model, hidden=hidden, lr=lr)
    yh = (p >= 0.5).astype(int)
    return dict(AUROC=roc_auc_score(yte, p),
                PR_AUC=average_precision_score(yte, p),
                recall=recall_score(yte, yh, zero_division=0),
                F1=f1_score(yte, yh, zero_division=0))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from best_single_baseline import load_xy, NUMERIC

    X, y, feat = load_xy()
    num_idx = [i for i, n in enumerate(feat) if n in NUMERIC]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    mu = Xtr[:, num_idx].mean(0); sd = Xtr[:, num_idx].std(0) + 1e-8
    Xtr = Xtr.copy(); Xte = Xte.copy()
    Xtr[:, num_idx] = (Xtr[:, num_idx] - mu) / sd; Xte[:, num_idx] = (Xte[:, num_idx] - mu) / sd
    nf = X.shape[1]

    def partition(y, k, alpha, seed=0):
        rng = np.random.default_rng(seed); silos = [[] for _ in range(k)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0]); pr = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
            for s, ch in enumerate(np.split(idx, cuts)): silos[s].extend(ch.tolist())
        return [np.sort(np.array(s, int)) for s in silos]

    silos = partition(ytr, 3, 0.5, 0)
    base = evaluate_run(Xtr, ytr, Xte, yte, silos, nf, no_churn(silos))
    print(f"baseline (no churn)         AUROC={base['AUROC']:.4f}")
    for rate in (0.3, 0.7):
        t = evaluate_run(Xtr, ytr, Xte, yte, silos, nf, transient_schedule(silos, rate))
        p = evaluate_run(Xtr, ytr, Xte, yte, silos, nf, permanent_schedule(silos, rate, 40))
        print(f"transient rate={rate}          AUROC={t['AUROC']:.4f} (d={t['AUROC']-base['AUROC']:+.4f})")
        print(f"permanent rate={rate}          AUROC={p['AUROC']:.4f} (d={p['AUROC']-base['AUROC']:+.4f})")
    # who leaves: rank silos by positive count
    pos_ct = [int(ytr[s].sum()) for s in silos]
    hi, lo = int(np.argmax(pos_ct)), int(np.argmin(pos_ct))
    ws_hi = evaluate_run(Xtr, ytr, Xte, yte, silos, nf, whole_silo_schedule(silos, [hi]))
    ws_lo = evaluate_run(Xtr, ytr, Xte, yte, silos, nf, whole_silo_schedule(silos, [lo]))
    print(f"whole-silo exit (pos-heavy #{hi}, {pos_ct[hi]} pos) AUROC={ws_hi['AUROC']:.4f} (d={ws_hi['AUROC']-base['AUROC']:+.4f})")
    print(f"whole-silo exit (pos-light #{lo}, {pos_ct[lo]} pos) AUROC={ws_lo['AUROC']:.4f} (d={ws_lo['AUROC']-base['AUROC']:+.4f})")
