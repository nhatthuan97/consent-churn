# Who Leaves Matters More Than How Many

Code and results for the paper **"Who Leaves Matters More Than How Many: The
Utility Cost of Revocable Patient Consent in Cross-Silo Federated Learning"**
(Nhan & Poudel, under review).

## What this measures

A growing literature adds blockchain consent layers to federated healthcare
ML so patients can revoke participation at any time — but nobody has measured
what that costs in model utility. We quantify it on 30-day readmission
prediction (UCI Diabetes 130-US-hospitals, ~102k real encounters, K=3 silos,
5 seeds), decomposing "consent churn" into three physically distinct regimes:

| Regime (70% churn / max stress) | ΔAUROC |
|---|---|
| Transient (withdraw, later rejoin) | −0.003 (≈ free) |
| Permanent, random | −0.007 (≈ free) |
| Permanent, biased (positives leave first) | −0.021 (~3×) |
| Whole-silo exit, positive-heavy (α=0.1) | **−0.059** (~6×) |
| Count-matched random control (same headcount) | −0.039 |

**Headline:** removing an entire positive-heavy silo costs −0.059 AUROC while
removing the *same number* of patients at random costs −0.039 — the damage is
carried by *who* leaves, not *how many* leave. A 200-run rerun with a
higher-capacity MLP client reproduces every delta within one standard
deviation: the cost structure is model-independent.

## Layout

```
code/
  data/                          # raw data (auto-downloaded on first run)
  experiment_setup.py            # SHARED: the split/standardise step and the
                                 #   Dirichlet silo partition. One definition,
                                 #   used by every experiment below.
  01_baseline_fullscale/
    best_single_baseline.py      # preprocessing + multi-model centralized benchmark
    federated_methods.py         # 5 FL aggregators; logreg + MLP flat-vector clients
    baseline_and_federated_methods.ipynb   # executed: ceiling/floor + 5 methods @ K=3
    baseline_full_scale_accuracy.ipynb     # exploratory sweeps
    best_single_baseline_results.json      # saved centralized ranking
  02_consent_churn/
    churn.py                     # consent-churn schedules (3 regimes + control) + runner
    consent_churn_study.ipynb    # executed: 3-regime study, 5 seeds, isolation test
    mlp_amplification_study.py   # runner: full churn grid, logreg AND MLP clients
    mlp_amplification.ipynb      # executed: model-capacity check results
    mlp_amplification_results.json  # all 200 per-run metrics
figures/                         # result figures used in the paper
```

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

The executed notebooks contain the studies with embedded results and
findings. Everything is pure NumPy and **deterministic to the seed**: rerunning
`mlp_amplification_study.py` regenerates `mlp_amplification_results.json`
**byte-for-byte identical** to the archived copy. Last verified 2026-09-18 on
Python 3.14.7 with NumPy 2.5.3 / pandas 3.0.5 / scikit-learn 1.9.1 — a stack
well ahead of the one the results were originally produced on.
