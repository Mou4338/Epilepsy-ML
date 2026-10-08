"""
Epileptic seizure detection with a neural network trained by PSO, APO, SBO
and the proposed Quantum Parity SBO (QP-SBO).

Datasets: UCI Epileptic Seizure Recognition (Bonn) and CHB-MIT scalp EEG,
          both prepared by preprocess.py.
Task    : binary - seizure (1) vs. non-seizure (0).

Usage  : python preprocess.py --uci ... --chbmit ...      (once)
         python run_experiment.py datasets/uci.npz
         python run_experiment.py datasets/chbmit.npz
"""
import sys, json, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from optimizers import Problem, ALGORITHMS

# ---------------------------------------------------------------- settings
N_HIDDEN     = 10      # hidden neurons
LB, UB       = -2.0, 2.0
N_POP        = 30
MAX_FE       = 15000   # same fitness-call budget for every optimizer (= 30 x 500)
N_RUNS       = 10
SEED         = 42
CURVE_POINTS = 100


# ---------------------------------------------------------------- data
def load_data(npz_path):
    """Load a dataset produced by preprocess.py (already split, standardised, PCA-reduced)."""
    d = np.load(npz_path, allow_pickle=True)
    return d["Xtr"], d["Xte"], d["ytr"], d["yte"], float(d["pca_var"])


# ---------------------------------------------------------------- network
N_IN = None            # set from the data (16 for UCI, 20 for CHB-MIT)


def n_weights(n_h=N_HIDDEN):
    return N_IN * n_h + n_h + n_h + 1


def unpack(w, n_h=N_HIDDEN):
    n_in, i = N_IN, 0
    W1 = w[i:i + n_in * n_h].reshape(n_in, n_h); i += n_in * n_h
    b1 = w[i:i + n_h]; i += n_h
    W2 = w[i:i + n_h]; i += n_h
    b2 = w[i]
    return W1, b1, W2, b2


def predict_proba(w, X):
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(X @ W1 + b1)
    z = h @ W2 + b2
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_loss(X, y):
    """Class-balanced binary cross-entropy (seizure class is 20% of data)."""
    pos_w = (1 - y.mean()) / y.mean()
    sw = np.where(y == 1, pos_w, 1.0)
    sw = sw / sw.mean()

    def loss(w):
        p = np.clip(predict_proba(w, X), 1e-9, 1 - 1e-9)
        return float(-np.mean(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))))
    return loss


def metrics(w, X, y):
    p = predict_proba(w, X)
    yp = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {
        "accuracy": accuracy_score(y, yp),
        "precision": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, yp),
        "auc": roc_auc_score(y, p),
        "cm": [int(tn), int(fp), int(fn), int(tp)],
    }


# ---------------------------------------------------------------- experiment
def main(path):
    ds = os.path.splitext(os.path.basename(path))[0]
    Xtr, Xte, ytr, yte, var = load_data(path)
    global N_IN
    N_IN = Xtr.shape[1]
    print(f"train {Xtr.shape}, test {Xte.shape}, seizure share {ytr.mean():.3f}, "
          f"PCA variance kept {var:.3f}, weights {n_weights()}")
    loss = make_loss(Xtr, ytr)
    results = {"meta": {"dataset": ds, "n_train": len(ytr), "n_test": len(yte), "pca_var": float(var),
                        "dim": n_weights(), "n_in": N_IN, "n_hidden": N_HIDDEN, "max_fe": MAX_FE, "n_pop": N_POP, "runs": N_RUNS},
               "algos": {}}
    for name, algo in ALGORITHMS.items():
        runs = []
        for r in range(N_RUNS):
            rng = np.random.default_rng(SEED + 1000 * r)
            prob = Problem(loss, n_weights(), LB, UB, MAX_FE)
            t0 = time.time()
            algo(prob, rng, n_pop=N_POP)
            m = metrics(prob.best_x, Xte, yte)
            m.update(train_loss=prob.best_f, time=time.time() - t0,
                     train_acc=metrics(prob.best_x, Xtr, ytr)["accuracy"],
                     curve=prob.curve(CURVE_POINTS).tolist())
            runs.append(m)
            print(f"{name:7s} run {r+1:2d}: loss {prob.best_f:.4f}  acc {m['accuracy']:.4f}  "
                  f"f1 {m['f1']:.4f}  auc {m['auc']:.4f}  {m['time']:.1f}s")
        results["algos"][name] = runs

    # Wilcoxon signed-rank: QP-SBO vs each baseline on final training loss
    q = [r["train_loss"] for r in results["algos"]["QP-SBO"]]
    results["wilcoxon"] = {}
    for name in ["PSO", "APO", "SBO"]:
        b = [r["train_loss"] for r in results["algos"][name]]
        results["wilcoxon"][name] = float(wilcoxon(q, b).pvalue)
    out = f"results_{ds}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
