"""Best single (centralized, pooled) baseline for early-readmission prediction.

Searches several model families with per-model class-imbalance handling and
selects by threshold-free metrics (AUROC, PR-AUC) via 5-fold stratified CV, so
the ranking is not an artifact of the 0.5 decision threshold. For the winning
model it also reports a tuned operating point.

Context: published state of the art on this dataset (30-day readmission) is
~0.667 AUROC (XGBoost) to ~0.70 (CatBoost) -- Liu et al. 2024, and the 2025
EHR-ML study -- so a well-tuned baseline here is expected to land ~0.67-0.70,
not higher. The dataset signal ceiling is low; this benchmark documents it.

Run:  conda run -n thesis python 01_baseline_fullscale/best_single_baseline.py
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "diabetes_raw.csv"
SEED = 0

# --------------------------------------------------------------------------- #
# Preprocessing (self-contained; mirrors the baseline notebook)
# --------------------------------------------------------------------------- #
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"]
DROP = ["weight", "payer_code", "examide", "citoglipton"]
DIAG = ["diag_1", "diag_2", "diag_3"]


def _icd9(code):
    if code is None or (isinstance(code, float) and np.isnan(code)): return "None"
    s = str(code).strip()
    if s in ("", "nan"):            return "None"
    if s.startswith(("V", "E")):    return "Other"
    if s.startswith("250"):         return "Diabetes"
    try: v = float(s)
    except ValueError:              return "Other"
    if 390 <= v <= 459 or v == 785: return "Circulatory"
    if 460 <= v <= 519 or v == 786: return "Respiratory"
    if 520 <= v <= 579 or v == 787: return "Digestive"
    if 800 <= v <= 999:             return "Injury"
    if 710 <= v <= 739:             return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "Genitourinary"
    if 140 <= v <= 239:             return "Neoplasms"
    return "Other"


def _collapse(s, min_frac=0.01):
    f = s.value_counts(normalize=True)
    return s.where(s.isin(set(f[f >= min_frac].index)), "Other")


def load_xy():
    df = pd.read_csv(DATA, low_memory=False)
    y = (df["readmitted"].astype(str) == "<30").astype(np.int64).to_numpy()
    d = df.drop(columns=["readmitted"] + [c for c in DROP if c in df])
    for c in DIAG:
        d[c] = d[c].map(_icd9)
    d["race"] = d["race"].fillna("Unknown")
    d["medical_specialty"] = _collapse(d["medical_specialty"].fillna("Missing").astype(str))
    d["max_glu_serum"] = d["max_glu_serum"].fillna("None")
    d["A1Cresult"] = d["A1Cresult"].fillna("None")
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        d[c] = _collapse(d[c].astype(str))
    num = [c for c in NUMERIC if c in d]
    cat = [c for c in d.columns if c not in num]
    X = pd.concat([d[num].astype(np.float32),
                   pd.get_dummies(d[cat].astype(str), prefix_sep="=", dtype=np.float32)],
                  axis=1)
    return X.to_numpy(np.float32), y, list(X.columns)


# --------------------------------------------------------------------------- #
# Model zoo (each with its own imbalance handling)
# --------------------------------------------------------------------------- #
def build_models(pos_weight: float) -> dict:
    return {
        "LogReg (balanced)": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced"))]),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "HistGradBoost (balanced)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            class_weight="balanced", random_state=SEED),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=SEED),
        "LightGBM (balanced)": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=SEED, verbose=-1),
    }


def cv_evaluate(model, X, y, skf):
    """Manual fold loop: collect out-of-fold probabilities + per-fold AUROC."""
    oof = np.zeros(len(y))
    fold_auroc, fold_ap = [], []
    for tr, te in skf.split(X, y):
        m = clone_fit(model, X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_auroc.append(roc_auc_score(y[te], p))
        fold_ap.append(average_precision_score(y[te], p))
    return oof, np.array(fold_auroc), np.array(fold_ap)


def clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[k]), float(f1[k])


def benchmark_centralized(X, y, n_splits=5, seed=SEED, verbose=True):
    """5-fold CV over every model; return (ranking_df, oof_probabilities)."""
    pos = y.mean()
    pos_weight = (1 - pos) / pos
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = build_models(pos_weight)
    rows, oof_store = [], {}
    for name, model in models.items():
        t = time.time()
        oof, fa, fap = cv_evaluate(model, X, y, skf)
        oof_store[name] = oof
        rows.append({"model": name,
                     "AUROC": roc_auc_score(y, oof), "AUROC_std": fa.std(),
                     "PR_AUC": average_precision_score(y, oof), "PR_AUC_std": fap.std(),
                     "sec": time.time() - t})
        if verbose:
            print(f"  {name:<28s} AUROC={rows[-1]['AUROC']:.4f}±{fa.std():.4f}  "
                  f"PR-AUC={rows[-1]['PR_AUC']:.4f}  [{time.time()-t:.0f}s]")
    res = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    return res, oof_store


def main():
    t0 = time.time()
    X, y, feat = load_xy()
    pos = y.mean()
    print(f"data {X.shape}  positive rate {pos:.4f}  scale_pos_weight {(1-pos)/pos:.2f}")
    print(f"reference: base-rate PR-AUC = {pos:.3f}; published SOTA AUROC ~0.667-0.70\n")

    res, oof_store = benchmark_centralized(X, y)
    print("\n=== MODEL RANKING (5-fold CV, sorted by AUROC) ===")
    print(res[["model", "AUROC", "AUROC_std", "PR_AUC"]].round(4).to_string(index=False))

    # Winner: operating points -------------------------------------------------
    best = res.iloc[0]["model"]
    p = oof_store[best]
    thr, f1_best = best_f1_threshold(y, p)
    yh_50 = (p >= 0.5).astype(int)
    yh_t = (p >= thr).astype(int)
    print(f"\n=== BEST MODEL: {best} ===")
    print(f"  AUROC  = {roc_auc_score(y, p):.4f}   PR-AUC = {average_precision_score(y, p):.4f}")
    print(f"  @0.50 threshold : recall={recall_score(y, yh_50):.3f}  "
          f"precision={precision_score(y, yh_50, zero_division=0):.3f}  F1={f1_score(y, yh_50):.3f}")
    print(f"  @{thr:.3f} (F1-opt): recall={recall_score(y, yh_t):.3f}  "
          f"precision={precision_score(y, yh_t):.3f}  F1={f1_score(y, yh_t):.3f}")

    out = HERE / "best_single_baseline_results.json"
    with open(out, "w") as f:
        json.dump({"ranking": res.to_dict(orient="records"),
                   "best_model": best,
                   "best_threshold": thr,
                   "n": int(len(y)), "n_features": X.shape[1],
                   "positive_rate": float(pos)}, f, indent=2)
    print(f"\nsaved -> {out}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
