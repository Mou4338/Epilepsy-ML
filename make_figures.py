"""Figures and summary table from a results file.
Usage: python make_figures.py results_uci.json   (figures saved as figures/uci_*.png)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1]))
DS = R["meta"]["dataset"]
TITLE = {"uci": "UCI (Bonn)", "chbmit": "CHB-MIT"}.get(DS, DS)
os.makedirs("figures", exist_ok=True)
OUT = lambda f: os.path.join("figures", f"{DS}_{f}")
A = R["algos"]
NAMES = ["PSO", "APO", "SBO", "QP-SBO"]
COL = {"PSO": "#2a78d6", "APO": "#eb6834", "SBO": "#1baf7a", "QP-SBO": "#eda100"}
MARK = {"PSO": "o", "APO": "s", "SBO": "^", "QP-SBO": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
fe = np.linspace(0, R["meta"]["max_fe"], len(A["PSO"][0]["curve"]))


def curves(n):
    return np.array([r["curve"] for r in A[n]])


# Fig 1 - convergence of all four optimizers
fig, ax = plt.subplots(figsize=(7, 4.2))
for n in NAMES:
    c = curves(n).mean(0)
    ax.plot(fe, c, color=COL[n], lw=2, marker=MARK[n], markevery=12, ms=6, label=n)
ax.set_yscale("log")
ax.set_xlabel("Function evaluations")
ax.set_ylabel("Training loss (weighted BCE, log scale)")
ax.set_title(f"{TITLE}: mean convergence over {R['meta']['runs']} runs", loc="left", color=INK)
ax.legend(frameon=False)
fig.savefig(OUT("fig1_convergence_all.png")); plt.close(fig)

# Fig 2 - SBO vs QP-SBO (mean with min-max band)
fig, ax = plt.subplots(figsize=(7, 4.2))
for n in ["SBO", "QP-SBO"]:
    C = curves(n)
    ax.fill_between(fe, C.min(0), C.max(0), color=COL[n], alpha=0.18, lw=0)
    ax.plot(fe, C.mean(0), color=COL[n], lw=2, marker=MARK[n], markevery=12, ms=6,
            label=f"{n} (mean, band = min-max)")
ax.set_yscale("log")
ax.set_xlabel("Function evaluations")
ax.set_ylabel("Training loss (log scale)")
ax.set_title(f"{TITLE}: SBO vs Quantum Parity SBO", loc="left", color=INK)
ax.legend(frameon=False)
fig.savefig(OUT("fig2_sbo_vs_qpsbo.png")); plt.close(fig)

# Fig 3 - distribution of final training loss
fig, ax = plt.subplots(figsize=(6.5, 4))
data = [[r["train_loss"] for r in A[n]] for n in NAMES]
bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK, "lw": 1.5})
for patch, n in zip(bp["boxes"], NAMES):
    patch.set_facecolor(COL[n]); patch.set_alpha(0.75); patch.set_edgecolor(INK2)
ax.set_xticks(range(1, 5), NAMES)
ax.set_ylabel("Final training loss")
ax.set_title(f"{TITLE}: final loss across {R['meta']['runs']} runs", loc="left", color=INK)
ax.grid(axis="x", visible=False)
fig.savefig(OUT("fig3_boxplot_loss.png")); plt.close(fig)

# Fig 4 - SBO vs QP-SBO test metrics (grouped bars)
KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
LAB = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(KEYS)); w = 0.2
for i, n in enumerate(NAMES):
    m = [np.mean([r[k] for r in A[n]]) for k in KEYS]
    s = [np.std([r[k] for r in A[n]]) for k in KEYS]
    ax.bar(x + (i - 1.5) * w, m, w * 0.9, yerr=s, color=COL[n], label=n,
           error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
ax.set_xticks(x, LAB)
lo = min(np.mean([r[k] for r in A[n]]) for n in NAMES for k in KEYS)
ax.set_ylim(max(0, lo - 0.08), 1.0)
ax.set_ylabel("Score (test set, mean ± sd)")
ax.set_title(f"{TITLE}: test-set performance", loc="left", color=INK)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
fig.savefig(OUT("fig4_test_metrics.png")); plt.close(fig)

# Fig 5 - confusion matrix of the best QP-SBO run
best = min(A["QP-SBO"], key=lambda r: r["train_loss"])
tn, fp, fn, tp = best["cm"]
M = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(M, cmap="Blues")
ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=13,
                color="white" if M[i, j] > M.max() / 2 else INK)
ax.set_xticks([0, 1], ["Non-seizure", "Seizure"]); ax.set_yticks([0, 1], ["Non-seizure", "Seizure"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"{TITLE}: QP-SBO confusion matrix", loc="left", color=INK, fontsize=10)
fig.savefig(OUT("fig5_confusion_qpsbo.png")); plt.close(fig)

# Summary table
rows = []
for n in NAMES:
    g = lambda k: (np.mean([r[k] for r in A[n]]), np.std([r[k] for r in A[n]]))
    rows.append({"algo": n, **{k: g(k) for k in KEYS + ["train_loss", "train_acc", "time"]},
                 "best_loss": min(r["train_loss"] for r in A[n])})
json.dump({"rows": rows, "wilcoxon": R["wilcoxon"], "meta": R["meta"], "best_qpsbo_cm": best["cm"]},
          open(f"summary_{DS}.json", "w"), indent=1)
for r in rows:
    print(r["algo"], {k: (round(v[0], 4), round(v[1], 4)) if isinstance(v, tuple) else v
                      for k, v in r.items() if k != "algo"})
print("Wilcoxon p (QP-SBO vs):", R["wilcoxon"])
