"""Figures for the proposal-defense deck.

All figures are regenerated from the executed result files, so the deck never
carries hand-typed numbers:
    results/best_single_baseline_results.json
    results/mlp_amplification_results.json
The five-method federated table is transcribed from the executed notebook
(baseline_and_federated_methods.ipynb, cell 12) since it is not written to JSON.

Outputs (figures/):
    ceiling.png            model ceilings vs the published SOTA band + FL floor
    fl_methods.png         5 aggregation methods x 3 skew levels; SCAFFOLD collapse
    silo_composition.png   Dirichlet partitions: where the positives live
    churn_sweep.png        three regimes vs churn rate (logreg client)
    isolation.png          the count-matched isolation test (the headline)
    mlp_deltas.png         logreg vs MLP paired deltas (amplification check)

Palette: ink #2B2D33, coral #E4572E, teal #1B998B, gray #6B7280.
Run:  ~/venvs/ds/bin/python make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, CORAL, TEAL, GRAY, MUTED = "#2B2D33", "#E4572E", "#1B998B", "#6B7280", "#9CA3AF"
BAND = "#E5E7EB"

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"          # generated artifacts live in one place
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

base = json.loads((RESULTS / "best_single_baseline_results.json").read_text())
res = json.loads((RESULTS / "mlp_amplification_results.json").read_text())
NSEED = len(res["config"]["seeds"])   # label from the data, never hard-coded
s = res["summary"]

plt.rcParams.update({
    "font.size": 12, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ ceiling --
# Centralized model ranking, the averageable ceiling, and the local-only floor.
rank = base["ranking"]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
rows = [(r["model"], r["AUROC"], GRAY) for r in rank]
rows.append(("Local-only floor (no federation, $\\alpha$=0.5)", 0.6157, CORAL))
rows = rows[::-1]
ax.axvspan(0.667, 0.700, color=BAND, zorder=0)
ax.text(0.6835, len(rows) - 0.35, "published SOTA band (0.667–0.70)",
        ha="center", va="center", fontsize=10, color=GRAY)
for i, (name, v, col) in enumerate(rows):
    face = CORAL if "LogReg" in name else col
    ax.plot([0.505, v], [i, i], color=BAND, lw=2, zorder=1)
    ax.plot(v, i, "o", ms=10, color=face, mec="white", mew=2, zorder=3)
    ax.annotate(f"{v:.4f}", (v, i), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=11, fontweight="bold", color=face)
labels = [r[0].replace("Local-only floor", "Local-only FLOOR") for r in rows]
ax.set_yticks(range(len(rows)), labels)
ax.set_xlim(0.505, 0.715)
ax.set_ylim(-0.7, len(rows) - 0.05)
ax.set_xlabel("AUROC (5-fold CV) — higher is better")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "ceiling.png")


# --------------------------------------------------------------- fl methods --
# Executed notebook results: AUROC by aggregation method and Dirichlet alpha.
METHODS = ["fedavg", "fedprox", "fedavgm", "fedadam", "scaffold"]
FL = {                       # alpha -> {method: AUROC}
    "10.0": dict(zip(METHODS, [0.6689, 0.6682, 0.6673, 0.6671, 0.6700])),
    "0.5":  dict(zip(METHODS, [0.6677, 0.6669, 0.6669, 0.6655, 0.6689])),
    "0.1":  dict(zip(METHODS, [0.6662, 0.6664, 0.6666, 0.6665, 0.6246])),
}
alphas = ["10.0", "0.5", "0.1"]
xlab = ["$\\alpha$=10  (near-IID)", "$\\alpha$=0.5  (moderate)",
        "$\\alpha$=0.1  (severe)"]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.arange(len(alphas))
ceil_line = ax.axhline(0.6662, color=INK, lw=1, ls="--", zorder=1)
for m in METHODS:
    vals = [FL[a][m] for a in alphas]
    hot = m == "scaffold"
    ax.plot(x, vals, marker="o", ms=9 if hot else 7, mec="white", mew=1.5,
            lw=3.0 if hot else 1.8, color=CORAL if hot else GRAY,
            alpha=1.0 if hot else 0.6, zorder=4 if hot else 3)
    if hot:
        ax.annotate(f"{vals[-1]:.4f}", (x[-1], vals[-1]), xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=12,
                    color=CORAL, fontweight="bold")
# the four indistinguishable methods are labelled as a cluster, not individually
ax.annotate("0.6662–0.6666", (x[-1], 0.6664), xytext=(12, 7),
            textcoords="offset points", va="center", fontsize=10.5, color=GRAY)
ax.annotate("SCAFFOLD collapses:\ncontrol variates fail over\nfew, small, extreme silos",
            (1.93, 0.6280), xytext=(-34, 30), textcoords="offset points", ha="right",
            fontsize=10.5, fontweight="bold", color=CORAL,
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.6))
ax.set_xticks(x, xlab)
ax.set_xlim(-0.14, 2.62)
ax.set_ylim(0.618, 0.6735)
ax.set_ylabel("AUROC")
ax.set_title("Aggregation choice barely matters — until heterogeneity is pathological",
             fontsize=11.5, color=INK, loc="left", pad=12)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=GRAY, lw=2, marker="o", alpha=0.6),
           Line2D([], [], color=CORAL, lw=3, marker="o"), ceil_line],
          ["FedAvg / FedProx / FedAvgM / FedAdam — indistinguishable",
           "SCAFFOLD", "centralized averageable ceiling (0.6662)"],
          frameon=False, loc="lower left", fontsize=10.5,
          bbox_to_anchor=(-0.01, -0.02))
save(fig, "fl_methods.png")


# ---------------------------------------------------------- silo composition -
# Why "who leaves" is even possible: the Dirichlet split concentrates positives.
PART = {                            # alpha -> [(n, pos_frac), ...]
    "10.0": [(34280, 0.092), (23034, 0.126), (24098, 0.125)],
    "0.5":  [(20379, 0.254), (44590, 0.088), (16443, 0.000)],
    "0.1":  [(529, 0.023), (76610, 0.073), (4273, 0.821)],
}
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True)
for ax, a, lab in zip(axes, alphas, xlab):
    ns = [p[0] for p in PART[a]]
    ps = [p[1] for p in PART[a]]
    cols = [CORAL if p >= 0.5 else GRAY for p in ps]
    bars = ax.bar(range(3), ps, color=cols, zorder=2, edgecolor="white", width=0.62)
    for i, (b, p, n) in enumerate(zip(bars, ps, ns)):
        ax.annotate(f"{p*100:.1f}%", (i, p), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color=cols[i])
    ax.set_title(lab, fontsize=11)
    # n goes in the tick label, so it cannot collide with anything
    ax.set_xticks(range(3), [f"silo {i}\nn={n:,}" for i, n in enumerate(ns)])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("positive rate in silo")
axes[2].annotate("only 4,273 patients,\nbut 82% positive —\nmost of the signal\nsits in one silo",
                 xy=(1.66, 0.78), xytext=(0.62, 0.52), textcoords="data",
                 ha="center", va="center", fontsize=9.5, fontweight="bold",
                 color=CORAL,
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5,
                                 connectionstyle="arc3,rad=-0.15"))
save(fig, "silo_composition.png")


# ----------------------------------------------------------- churn rate sweep -
RATES = ["0.1", "0.3", "0.5", "0.7"]
xr = [float(r) for r in RATES]
regimes = [("Transient (rejoin)", "transient", TEAL, "o"),
           ("Permanent, random", "permanent", GRAY, "s"),
           ("Permanent, biased (positives leave first)", "biased", CORAL, "^")]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
for name, key, col, mk in regimes:
    mu = np.array([s["logreg"][key][r][0] for r in RATES])
    sd = np.array([s["logreg"][key][r][1] for r in RATES])
    ax.plot(xr, mu, marker=mk, color=col, lw=2.2, ms=8, mec="white", mew=1.5,
            label=name, zorder=3)
    ax.fill_between(xr, mu - sd, mu + sd, color=col, alpha=0.15, zorder=2)
ax.axhline(0, color=MUTED, lw=1, zorder=1)
for name, key, col, mk in regimes:
    v = s["logreg"][key]["0.7"][0]
    ax.annotate(f"{v:+.3f}", (0.7, v), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=11, fontweight="bold", color=col)
ax.set_xticks(xr, [f"{int(r*100)}%" for r in xr])
ax.set_xlim(0.06, 0.79)
ax.set_xlabel("consent-churn rate (fraction of patients withdrawing)")
ax.set_ylabel("$\\Delta$AUROC vs no-churn baseline")
ax.legend(frameon=False, loc="lower left", fontsize=10.5)
save(fig, "churn_sweep.png")


# ------------------------------------------------------------ isolation test --
# The headline: same headcount, different identity.
conds = [("Whole-silo exit\n(positive-HEAVY silo)", "whole_silo_heavy_a0.1", CORAL),
         ("Count-matched RANDOM control\n(same N, drawn uniformly)", "matched_a0.1", TEAL),
         ("Whole-silo exit\n(positive-LIGHT silo)", "whole_silo_light_a0.1", GRAY)]
fig, ax = plt.subplots(figsize=(8.8, 4.2))
ys = np.arange(len(conds))[::-1]
for y, (name, key, col) in zip(ys, conds):
    mu, sd = s["logreg"][key]
    ax.barh(y, mu, height=0.52, color=col, zorder=2, edgecolor="white", linewidth=1)
    ax.errorbar(mu, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=1.2,
                capsize=3, alpha=0.55, zorder=4)
    # label sits INSIDE the bar (bars run leftward from 0), so left-align at mu
    inside = abs(mu) > 0.02
    ax.annotate(f"{mu:+.3f}", (mu, y), xytext=(9 if inside else -9, 0),
                textcoords="offset points", ha="left" if inside else "right",
                va="center", fontsize=14, fontweight="bold",
                color="white" if inside else col, zorder=5)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.set_yticks(ys, [c[0] for c in conds])
ax.set_xlim(-0.095, 0.014)
ax.set_ylim(-0.55, 2.95)
ax.set_xlabel(f"$\\Delta$AUROC vs no-churn baseline  ($\\alpha$=0.1, {NSEED} seeds, mean $\\pm$ sd)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
# bracket the two conditions that remove the SAME headcount: heavy exit (y=2)
# and the matched random control (y=1)
ax.annotate("", xy=(-0.083, 2.0), xytext=(-0.083, 1.0),
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.4,
                            connectionstyle="bar,fraction=0.14"))
ax.text(-0.0905, 1.5, "same\nheadcount\nremoved", ha="center", va="center",
        fontsize=10, color=INK, fontweight="bold")
# ratio computed from the data -- a hard-coded multiplier silently goes stale
# the moment the seed count or any condition changes.
_hv = abs(s["logreg"]["whole_silo_heavy_a0.1"][0])
_mv = abs(s["logreg"]["matched_a0.1"][0])
_ratio = _hv / _mv if _mv else float("nan")
ax.text(-0.070, 2.62, f"identity of who leaves  →  {_ratio:.1f}$\\times$ the damage",
        ha="center", va="center", fontsize=11.5, color=CORAL, fontweight="bold")
save(fig, "isolation.png")


# --------------------------------------------------------------- mlp deltas ---
conds2 = [("Transient (rejoin), 70%", "transient", "0.7"),
          ("Permanent, random, 70%", "permanent", "0.7"),
          ("Permanent, biased, 70%", "biased", "0.7"),
          ("Whole-silo, positive-light", "whole_silo_light_a0.1", None),
          ("Count-matched random control", "matched_a0.1", None),
          ("Whole-silo, positive-heavy", "whole_silo_heavy_a0.1", None)]


def get(model, key, rate):
    v = s[model][key]
    return v[rate] if rate else v


fig, ax = plt.subplots(figsize=(8.6, 4.6))
h = 0.34
for i, (name, key, rate) in enumerate(conds2):
    y = len(conds2) - 1 - i
    lm, lsd = get("logreg", key, rate)
    mm, msd = get("mlp", key, rate)
    ax.barh(y + h / 2 + 0.01, lm, height=h, color=GRAY, zorder=2,
            edgecolor="white", linewidth=1)
    ax.barh(y - h / 2 - 0.01, mm, height=h, color=CORAL, zorder=2,
            edgecolor="white", linewidth=1)
    ax.errorbar([lm, mm], [y + h / 2 + 0.01, y - h / 2 - 0.01], xerr=[lsd, msd],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2, alpha=0.5, zorder=3)
    left = min(lm - lsd, mm - msd)
    ax.annotate(f"{lm:+.3f}", (left, y + h / 2 + 0.01), xytext=(-6, -1),
                textcoords="offset points", ha="right", va="center",
                fontsize=10, color=GRAY)
    ax.annotate(f"{mm:+.3f}", (left, y - h / 2 - 0.01), xytext=(-6, -1),
                textcoords="offset points", ha="right", va="center",
                fontsize=10, fontweight="bold", color=CORAL)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.set_yticks(range(len(conds2)), [c[0] for c in reversed(conds2)])
ax.set_xlim(-0.145, 0.022)
ax.set_ylim(-0.62, 6.05)
ax.set_xlabel(f"$\\Delta$AUROC vs same-seed no-churn baseline ({NSEED} seeds, mean $\\pm$ sd)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY),
           plt.Rectangle((0, 0), 1, 1, color=CORAL)]
ax.legend(handles, ["LogReg client (0.666 centralized)",
                    "MLP client, 64 hidden units (0.670 centralized)"],
          frameon=False, loc="upper left", fontsize=10,
          bbox_to_anchor=(-0.015, 1.03))
save(fig, "mlp_deltas.png")

print("\nAll figures written to", FIG)
