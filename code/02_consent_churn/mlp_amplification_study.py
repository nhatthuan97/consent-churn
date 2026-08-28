"""MLP amplification check for the consent-churn study.

The churn study measured modest utility deltas on a class-weighted logistic
regression client. Open question (flagged in the study's Findings and in the
dissertation): are those magnitudes real, or an artifact of a saturated linear
model? This script reruns the full churn grid with BOTH clients — the original
logreg and a one-hidden-layer MLP (64 ReLU units, lr 0.05, chosen because it
centralizes at AUROC 0.670 > logreg's 0.666) — on identical partitions and
schedules, so every delta is an apples-to-apples pair.

Grid per model (100 runs each; FedAvg, class-weighted, 40 rounds, K=3, 5 seeds):
  baselines      no-churn at alpha in {0.5, 0.1}
  experiment 1   transient / permanent(random) / permanent(biased) at
                 rates {0.1, 0.3, 0.5, 0.7}, alpha=0.5
  experiment 2   whole-silo exit (pos-heavy / pos-light), alpha in {0.5, 0.1}
  experiment 3   count-matched random control for the pos-heavy exit

Usage:  conda run -n thesis python mlp_amplification_study.py
Writes mlp_amplification_results.json (all per-run rows + config) and prints
the summary tables. Deterministic: same seeds -> same numbers on rerun.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")   # before numpy: keep workers lean

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_baseline_fullscale"))

from churn import (evaluate_run, no_churn, transient_schedule, permanent_schedule,
                   biased_permanent_schedule, whole_silo_schedule,
                   matched_random_schedule)

SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 40
RATES = [0.1, 0.3, 0.5, 0.7]
ALPHAS = [0.5, 0.1]
K = 3
MODELS = {"logreg": dict(model="logreg", hidden=32, lr=0.1),
          "mlp":    dict(model="mlp", hidden=64, lr=0.05)}
RESULTS = Path(__file__).resolve().parent / "mlp_amplification_results.json"

# set in _init_worker / main; inherited by fork
_DATA = {}


def _load_data():
    from sklearn.model_selection import train_test_split
    from best_single_baseline import load_xy, NUMERIC
    X, y, feat = load_xy()
    num_idx = [i for i, n in enumerate(feat) if n in NUMERIC]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=0)
    mu = Xtr[:, num_idx].mean(0); sd = Xtr[:, num_idx].std(0) + 1e-8
    Xtr = Xtr.copy(); Xte = Xte.copy()
    Xtr[:, num_idx] = (Xtr[:, num_idx] - mu) / sd
    Xte[:, num_idx] = (Xte[:, num_idx] - mu) / sd
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, nf=X.shape[1])


def partition(y, k, alpha, seed=0):
    rng = np.random.default_rng(seed); silos = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); pr = rng.dirichlet(alpha * np.ones(k))
        cuts = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
        for s, ch in enumerate(np.split(idx, cuts)): silos[s].extend(ch.tolist())
    return [np.sort(np.array(s, int)) for s in silos]


def _build_schedule(spec, silos, ytr, seed):
    """Schedules are closures (not picklable), so workers rebuild them from a
    plain tuple spec. Silo choice for whole-silo/matched ranks by positive count."""
    kind = spec[0]
    if kind == "none":
        return no_churn(silos), {}
    if kind == "transient":
        return transient_schedule(silos, spec[1], seed=seed), {"rate": spec[1]}
    if kind == "permanent":
        return permanent_schedule(silos, spec[1], ROUNDS, seed=seed), {"rate": spec[1]}
    if kind == "biased":
        return (biased_permanent_schedule(silos, ytr, spec[1], ROUNDS, seed=seed),
                {"rate": spec[1]})
    pos_ct = [int(ytr[s].sum()) for s in silos]
    hi, lo = int(np.argmax(pos_ct)), int(np.argmin(pos_ct))
    if kind == "whole_silo":
        sid = hi if spec[1] == "heavy" else lo
        return (whole_silo_schedule(silos, [sid]),
                {"which": spec[1], "n_removed": len(silos[sid]),
                 "pos_removed": pos_ct[sid]})
    if kind == "matched":
        removed = matched_random_schedule(silos, len(silos[hi]), seed=seed)
        return removed, {"n_removed": len(silos[hi])}
    raise ValueError(spec)


def _run_one(job):
    model_name, alpha, seed, spec = job
    d = _DATA
    silos = partition(d["ytr"], K, alpha, seed)
    sched, extra = _build_schedule(spec, silos, d["ytr"], seed)
    kw = MODELS[model_name]
    m = evaluate_run(d["Xtr"], d["ytr"], d["Xte"], d["yte"], silos, d["nf"],
                     sched, seed=seed, rounds=ROUNDS, **kw)
    return dict(model=model_name, alpha=alpha, seed=seed, regime=spec[0],
                **extra, **m)


def build_jobs():
    jobs = []
    for model in MODELS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                jobs.append((model, alpha, seed, ("none",)))
                jobs.append((model, alpha, seed, ("whole_silo", "heavy")))
                jobs.append((model, alpha, seed, ("whole_silo", "light")))
                jobs.append((model, alpha, seed, ("matched", "heavy")))
        for rate in RATES:
            for regime in ("transient", "permanent", "biased"):
                for seed in SEEDS:
                    jobs.append((model, 0.5, seed, (regime, rate)))
    return jobs


def summarize(rows):
    """Per-model deltas vs the same (alpha, seed) no-churn baseline."""
    base = {(r["model"], r["alpha"], r["seed"]): r for r in rows
            if r["regime"] == "none"}

    def deltas(sel):
        d = [r["AUROC"] - base[(r["model"], r["alpha"], r["seed"])]["AUROC"]
             for r in sel]
        return float(np.mean(d)), float(np.std(d))

    out = {}
    for model in MODELS:
        s = {}
        s["baseline_auroc"] = {str(a): float(np.mean(
            [base[(model, a, sd)]["AUROC"] for sd in SEEDS])) for a in ALPHAS}
        for regime in ("transient", "permanent", "biased"):
            s[regime] = {str(rate): deltas(
                [r for r in rows if r["model"] == model and r["regime"] == regime
                 and r["rate"] == rate]) for rate in RATES}
        for a in ALPHAS:
            for which in ("heavy", "light"):
                s[f"whole_silo_{which}_a{a}"] = deltas(
                    [r for r in rows if r["model"] == model
                     and r["regime"] == "whole_silo" and r["which"] == which
                     and r["alpha"] == a])
            s[f"matched_a{a}"] = deltas(
                [r for r in rows if r["model"] == model
                 and r["regime"] == "matched" and r["alpha"] == a])
        out[model] = s
    return out


def print_summary(s):
    lr, mlp = s["logreg"], s["mlp"]
    print(f"\nno-churn baseline AUROC   logreg {lr['baseline_auroc']}   "
          f"mlp {mlp['baseline_auroc']}")
    print("\n== Experiment 1: rate sweep (alpha=0.5), dAUROC mean (sd) ==")
    print(f"{'regime':<12s}{'rate':>6s}{'logreg':>18s}{'mlp':>18s}{'ratio':>8s}")
    for regime in ("transient", "permanent", "biased"):
        for rate in RATES:
            a, sa = lr[regime][str(rate)]; b, sb = mlp[regime][str(rate)]
            ratio = b / a if abs(a) > 1e-6 else float("nan")
            print(f"{regime:<12s}{rate:>6.1f}{a:>+11.4f} ({sa:.4f})"
                  f"{b:>+11.4f} ({sb:.4f}){ratio:>8.1f}")
    print("\n== Experiments 2-3: silo-level churn, dAUROC mean (sd) ==")
    print(f"{'condition':<26s}{'logreg':>18s}{'mlp':>18s}")
    for a in ALPHAS:
        for key in (f"whole_silo_heavy_a{a}", f"whole_silo_light_a{a}",
                    f"matched_a{a}"):
            x, sx = lr[key]; y, sy = mlp[key]
            print(f"{key:<26s}{x:>+11.4f} ({sx:.4f}){y:>+11.4f} ({sy:.4f})")


def main():
    global _DATA
    _DATA = _load_data()
    jobs = build_jobs()
    print(f"{len(jobs)} runs, {os.cpu_count()} cores", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, row in enumerate(ex.map(_run_one, jobs, chunksize=2)):
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} done", flush=True)
    summary = summarize(rows)
    RESULTS.write_text(json.dumps(dict(
        config=dict(seeds=SEEDS, rounds=ROUNDS, rates=RATES, alphas=ALPHAS, k=K,
                    models={k: v for k, v in MODELS.items()}),
        summary=summary, runs=rows), indent=1))
    print(f"\nwrote {RESULTS}")
    print_summary(summary)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
