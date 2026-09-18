"""MLP amplification check for the consent-churn study.

The churn study measured modest utility deltas on a class-weighted logistic
regression client. Open question (flagged in the study's Findings and in the
dissertation): are those magnitudes real, or an artifact of a saturated linear
model? This script reruns the full churn grid with BOTH clients — the original
logreg and a one-hidden-layer MLP (64 ReLU units, lr 0.05, chosen because it
centralizes at AUROC 0.670 > logreg's 0.666) — on identical partitions and
schedules, so every delta is an apples-to-apples pair.

Grid per model (600 runs each; FedAvg, class-weighted, 40 rounds, K=3, 30 seeds):
  baselines      no-churn at alpha in {0.5, 0.1}
  experiment 1   transient / permanent(random) / permanent(biased) at
                 rates {0.1, 0.3, 0.5, 0.7}, alpha=0.5
  experiment 2   whole-silo exit (pos-heavy / pos-light), alpha in {0.5, 0.1}
  experiment 3   count-matched random control for the pos-heavy exit

Usage:  ~/venvs/ds/bin/python mlp_amplification_study.py
Writes mlp_amplification_results.json (all per-run rows + config) and prints
the summary tables. Deterministic: same seeds -> same numbers on rerun.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")   # before numpy: keep workers lean

import json
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_baseline_fullscale"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from churn import (evaluate_run, no_churn, transient_schedule, permanent_schedule,
                   biased_permanent_schedule, whole_silo_schedule,
                   matched_random_schedule)
from experiment_setup import load_split_standardize, partition, results_path

# 30 seeds, not 5. The headline whole-silo vs count-matched contrast has a
# paired sd of roughly 0.046; at n=5 that is ~2.5x the effect and the paired
# test cannot resolve it (p~0.43). n=30 brings it to p~1e-4. The grid costs
# about 8 minutes, so there is no reason to run underpowered.
SEEDS = list(range(30))
ROUNDS = 40
RATES = [0.1, 0.3, 0.5, 0.7]
ALPHAS = [0.5, 0.1]
K = 3
MODELS = {"logreg": dict(model="logreg", hidden=32, lr=0.1),
          "mlp":    dict(model="mlp", hidden=64, lr=0.05)}
RESULTS = results_path("mlp_amplification_results.json")

# set in _init_worker / main; inherited by fork
_DATA = {}


def _load_data():
    return load_split_standardize()


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


def paired_tests(rows):
    """The isolation test, done as a paired comparison rather than two means.

    whole-silo exit and its count-matched control share a seed, hence an
    identical partition and an identical number of patients removed, so they are
    paired observations and the difference must be tested per seed. Reporting
    only the two marginal means hides the pairing and overstates what a small
    number of seeds can support: the paired sd here is several times the effect,
    so n=5 leaves the central claim untestable.
    """
    from scipy import stats

    base = {(r["model"], r["alpha"], r["seed"]): r["AUROC"] for r in rows
            if r["regime"] == "none"}

    def series(model, alpha, pred):
        out = {}
        for r in rows:
            if r["model"] == model and r["alpha"] == alpha and pred(r):
                out[r["seed"]] = r["AUROC"] - base[(model, alpha, r["seed"])]
        return out

    res = {}
    for model in MODELS:
        for alpha in ALPHAS:
            ws = series(model, alpha, lambda r: r["regime"] == "whole_silo"
                        and r.get("which") == "heavy")
            mr = series(model, alpha, lambda r: r["regime"] == "matched")
            seeds = sorted(set(ws) & set(mr))
            if len(seeds) < 3:
                continue
            a = np.array([ws[s] for s in seeds]); b = np.array([mr[s] for s in seeds])
            diff = a - b
            t, pval = stats.ttest_rel(a, b)
            try:
                wp = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:
                wp = float("nan")
            lo, hi = stats.t.interval(0.95, len(diff) - 1,
                                      loc=diff.mean(), scale=stats.sem(diff))
            res[f"{model}_a{alpha}"] = dict(
                n=len(seeds),
                whole_silo_mean=float(a.mean()), whole_silo_sd=float(a.std(ddof=1)),
                matched_mean=float(b.mean()), matched_sd=float(b.std(ddof=1)),
                paired_diff=float(diff.mean()), paired_sd=float(diff.std(ddof=1)),
                t=float(t), p_ttest=float(pval), p_wilcoxon=wp,
                ci95=[float(lo), float(hi)],
                n_correct_direction=int((diff < 0).sum()))
    return res


def print_paired(pt):
    print("\n== Isolation test, paired by seed (the H2 claim) ==")
    print(f"{'condition':<16}{'n':>4}{'whole-silo':>13}{'matched':>12}"
          f"{'paired diff':>13}{'p (t)':>10}{'dir':>8}")
    for k, v in pt.items():
        print(f"{k:<16}{v['n']:>4}{v['whole_silo_mean']:>+13.4f}{v['matched_mean']:>+12.4f}"
              f"{v['paired_diff']:>+13.4f}{v['p_ttest']:>10.4f}"
              f"{v['n_correct_direction']:>5}/{v['n']}")
        print(f"{'':<16}95% CI on the paired difference "
              f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]   "
              f"Wilcoxon p={v['p_wilcoxon']:.4f}")


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
    # Workers read the module-level _DATA populated above. Python 3.14 changed
    # the Linux default start method to forkserver, which re-imports this module
    # and leaves _DATA empty; fork keeps the copy-on-write inheritance the
    # archived runs used (and avoids pickling the design matrix to each worker).
    with ProcessPoolExecutor(max_workers=10,
                             mp_context=mp.get_context("fork")) as ex:
        for i, row in enumerate(ex.map(_run_one, jobs, chunksize=2)):
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} done", flush=True)
    summary = summarize(rows)
    paired = paired_tests(rows)
    RESULTS.write_text(json.dumps(dict(
        config=dict(seeds=SEEDS, rounds=ROUNDS, rates=RATES, alphas=ALPHAS, k=K,
                    models={k: v for k, v in MODELS.items()}),
        summary=summary, paired_tests=paired, runs=rows), indent=1))
    print(f"\nwrote {RESULTS}")
    print_summary(summary)
    print_paired(paired)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
