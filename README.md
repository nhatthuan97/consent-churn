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

```bash
conda create -n thesis python=3.11 -y
conda run -n thesis pip install numpy pandas scikit-learn scipy matplotlib \
    ucimlrepo jupyterlab nbformat nbconvert ipykernel xgboost lightgbm
```

## Reproducing

The dataset auto-downloads to `code/data/` on first run. From
`code/01_baseline_fullscale/`:

```bash
conda run -n thesis python best_single_baseline.py   # centralized benchmark
conda run -n thesis python federated_methods.py      # 5 aggregators sanity sweep
```

From `code/02_consent_churn/`:

```bash
conda run -n thesis python churn.py                     # quick churn demo
conda run -n thesis python mlp_amplification_study.py   # full 200-run grid
```

The executed notebooks contain the studies with embedded results and
findings. Everything is pure NumPy and **deterministic to the seed**: in our
replication, rerunning stored configurations reproduces the archived JSON
results bit-exactly across NumPy/scikit-learn versions.
