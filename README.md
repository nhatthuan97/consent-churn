# Who Leaves Matters More Than How Many

Code and results for the paper **"Who Leaves Matters More Than How Many: The
Utility Cost of Revocable Patient Consent in Cross-Silo Federated Learning"**
(Nhan & Poudel, under review).

## Citing

Under review; no venue named until a decision. Cite as:

```bibtex
@misc{nhan2026wholeaves,
  author = {Nhan, Thuan and Poudel, Khem},
  title  = {Who Leaves Matters More Than How Many: The Utility Cost of
            Revocable Patient Consent in Cross-Silo Federated Learning},
  year   = {2026},
  note   = {Manuscript under review},
  howpublished = {\url{https://github.com/nhatthuan97/consent-churn}}
}
```

`references.bib` holds the works this study builds on, each verified against the
publisher of record rather than reconstructed from memory. `CITATION.cff` carries
the same metadata in machine-readable form.

## What this measures

A growing literature adds blockchain consent layers to federated healthcare
ML so patients can revoke participation at any time — but nobody has measured
what that costs in model utility. We quantify it on 30-day readmission
prediction (UCI Diabetes 130-US-hospitals, ~102k real encounters, K=3 silos,
**30 seeds**), decomposing "consent churn" into three physically distinct
regimes:

| Regime (70% churn / max stress) | ΔAUROC (sd) |
|---|---|
| Transient (withdraw, later rejoin) | −0.000 (0.005) — free |
| Permanent, random | −0.007 (0.006) |
| Permanent, biased (positives leave first) | −0.018 (0.017) — ~2.5× |
| Whole-silo exit, positive-heavy (α=0.1) | **−0.053** (0.045) |
| Whole-silo exit, positive-light (α=0.1) | −0.001 (0.018) |
| Count-matched random control (same headcount) | −0.015 (0.028) |

**Headline.** Removing an entire positive-heavy silo costs −0.053 AUROC, while
removing the *same number* of patients drawn at random costs −0.015 — **3.6×
less**. The damage is carried by *who* leaves, not *how many*.

The two conditions share a seed, so they share a partition and remove an
identical number of patients: they are paired observations, and the contrast is
tested per seed rather than read off two marginal means.

| Isolation test (α=0.1, logreg) | |
|---|---|
| Paired difference | **−0.038** (sd 0.046, n=30) |
| 95% CI | [−0.055, −0.021] — excludes zero |
| Paired t-test | t = −4.49, **p = 0.0001** |
| Wilcoxon signed-rank | **p = 0.0001** |
| Seeds in the predicted direction | 23/30 |

The effect also holds at milder skew (α=0.5: paired difference −0.008,
p = 0.0075) and under the higher-capacity client (MLP, α=0.1: −0.031,
p = 0.0002), so it is not an artifact of one skew level or one model class.

**Why 30 seeds.** The paired standard deviation here is roughly 0.046 — several
times the effect itself. At the 5 seeds used in an earlier draft the same
comparison gave p = 0.43 with only 3/5 seeds in the predicted direction: the
result was real but untestable. The grid costs about eight minutes, so there is
no reason to run it underpowered. `paired_tests()` in
`mlp_amplification_study.py` computes these statistics as part of the run.

**Model independence.** Rerunning the entire grid with a higher-capacity MLP
client on identical partitions and schedules reproduces every delta within one
standard deviation of the logistic-regression result — 0 of 18 conditions
exceed it. The cost structure is model-independent.

**A caveat worth stating.** At α=0.1 with K=3 the Dirichlet partition is
genuinely extreme: in 8 of 30 seeds the positive-heavy silo holds more than 80%
of the training set, so its departure removes most of the data. This is a real
property of severe skew rather than a bug, but it inflates variance, and it is
why the paired test matters. Restricting to seeds where the exiting silo holds
at most half the training data leaves the conclusion intact (n=21, paired
difference −0.041, p = 0.0002).

## Layout

```
code/
  data/                          # raw data (auto-downloaded on first run)
  experiment_setup.py            # SHARED: split/standardise, the Dirichlet silo
                                 #   partition, and where results/ lives.
  01_baseline_fullscale/
    best_single_baseline.py      # preprocessing + multi-model centralized benchmark
    federated_methods.py         # 5 FL aggregators; logreg + MLP flat-vector clients
    baseline_and_federated_methods.ipynb   # executed: ceiling/floor + 5 methods @ K=3
    baseline_full_scale_accuracy.ipynb     # exploratory sweeps
  02_consent_churn/
    churn.py                     # consent-churn schedules (3 regimes + control) + runner
    consent_churn_study.ipynb    # executed: 3-regime study, 5 seeds, isolation test
    mlp_amplification_study.py   # runner: full churn grid, logreg AND MLP clients
    mlp_amplification.ipynb      # executed: model-capacity check results
make_figs.py                     # rebuilds results/figures/ from the JSON
results/                         # EVERYTHING GENERATED, in one place
  best_single_baseline_results.json   # saved centralized ranking
  mlp_amplification_results.json      # all 200 per-run metrics
  figures/                       # result figures used in the paper
```

Scripts write only into `results/`; nothing generated is stored beside the code
that produced it, so a rerun refreshes every downstream artifact at once.


## Environment

A plain virtualenv is enough; `requirements.txt` pins the versions the results
were last verified against.

```bash
python3 -m venv ~/venvs/ds
~/venvs/ds/bin/pip install -r requirements.txt
```

**Python 3.14 note.** On Linux, Python 3.14 changed the default multiprocessing
start method from `fork` to `forkserver`. `mlp_amplification_study.py` shares
the preloaded design matrix with its workers through a module-level global, so
it explicitly requests a `fork` context; under `forkserver` the workers re-import
the module and see an empty global. Keep that context if you touch the runner.

## Reproducing

The dataset auto-downloads to `code/data/` on first run. From
`code/01_baseline_fullscale/`:

```bash
~/venvs/ds/bin/python best_single_baseline.py   # centralized benchmark
~/venvs/ds/bin/python federated_methods.py      # 5 aggregators sanity sweep
```

From `code/02_consent_churn/`:

```bash
~/venvs/ds/bin/python churn.py                     # quick churn demo
~/venvs/ds/bin/python mlp_amplification_study.py   # full 200-run grid
```

### Cost to reproduce

Everything is CPU-only NumPy; no GPU is needed and nothing runs for hours.
Measured on 16 cores (the grid uses 10 workers):

| Command | Wall clock |
|---|---|
| `best_single_baseline.py` | ~30 s |
| `mlp_amplification_study.py` (all 200 runs) | **~80 s** |
| `churn.py`, `federated_methods.py` | ~1-2 min each |

The dataset (~16 MB) downloads itself from the UCI repository the first time any
entry point loads it, so a clean checkout needs no manual data step.

### Verified reproduction (2026-09-18)

Rerun end to end on Python 3.14.7 / NumPy 2.5.3 / pandas 3.0.5 / scikit-learn
1.9.1 / XGBoost 3.4.1 / LightGBM 4.7.0:

| Artifact | Result |
|---|---|
| `mlp_amplification_results.json` (200 runs) | **byte-identical** to the archived copy |
| `federated_methods.py` | reproduces the flat method comparison and the SCAFFOLD collapse at alpha=0.1 (AUROC 0.6246) |
| `churn.py` | reproduces the regime ordering |
| `best_single_baseline_results.json` | reproduces except **XGBoost AUROC 0.676882 -> 0.676166** (-7.2e-04) under XGBoost 3.4.1 |

The XGBoost drift is a library-version effect, an order of magnitude below that
model's own CV standard deviation (+/-0.0064). It does not move the ranking, the
selected model, or the tuned threshold, all of which reproduce exactly. The
committed JSON is kept as the archived artifact rather than being overwritten.

This was checked the way a reviewer would: clone the repository fresh, build a
new virtualenv from `requirements.txt`, and run the grid. It downloaded the data
itself and regenerated `mlp_amplification_results.json` byte-for-byte identical
to the archived copy, on a software stack well ahead of the one the results were
originally produced on.

The executed notebooks carry their embedded results, so the studies and findings
can be read without running anything.
