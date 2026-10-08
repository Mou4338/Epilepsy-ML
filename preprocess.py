"""
Preprocessing and standardisation for both datasets.

  1. UCI Epileptic Seizure Recognition (Bonn EEG, 11,500 x 178)
  2. CHB-MIT Scalp EEG (PhysioNet), any set of patients

Both datasets end up in the same format, ready for run_experiment.py:
    datasets/<name>.npz  ->  Xtr, Xte, ytr, yte (PCA-reduced, standardised),
                             Xtr_full, Xte_full (standardised, before PCA),
                             feature_names, groups_tr, groups_te

Key design choices (also described in the paper)
  * Leakage-free split: segments are split by RECORDING, never by segment,
    so pieces of one recording never appear in both train and test.
    UCI    -> group = original Bonn recording (decoded from the ID column)
    CHB-MIT-> group = EDF file
  * Standardisation (z-score) and PCA are fitted on the training set only and
    then applied to the test set.
  * Labels: 1 = seizure, 0 = non-seizure.

Needs only numpy, scipy, pandas and scikit-learn (no MNE / pyEDFlib):
EDF files and the PhysioNet *.edf.seizures annotations are read directly.

Usage
  python preprocess.py --uci "Epileptic Seizure Recognition.csv"
  python preprocess.py --chbmit path/to/chb-mit --patients chb01 chb02 chb03 chb05 chb08
  (both flags can be given together)
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
N_COMPONENTS = 20          # PCA size when a dataset has more than 20 features
N_SPLITS = 3               # 3 folds -> about 1/3 of recordings held out for test
OUT_DIR = "datasets"

# CHB-MIT settings
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]            # 18 bipolar channels present in almost every file
FS = 256
WIN_SEC = 4                               # window length
SEIZURE_STEP_SEC = 1                      # 75 % overlap inside seizures (more minority samples)
NORMAL_STEP_SEC = 4                       # no overlap elsewhere
NORMAL_PER_SEIZURE = 4                    # keep 4 non-seizure windows per seizure window
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


# =============================================================== shared helpers
def leakage_free_split(y, groups):
    """Hold out whole recordings, keeping the seizure share similar in both parts."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    assert not set(groups[tr]) & set(groups[te])
    return tr, te


def standardise_and_save(name, X, y, groups, feature_names):
    tr, te = leakage_free_split(y, groups)
    scaler = StandardScaler().fit(X[tr])                    # fitted on train only
    Xtr_full, Xte_full = scaler.transform(X[tr]), scaler.transform(X[te])
    if X.shape[1] > N_COMPONENTS:                           # reduce only when needed
        pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Xtr_full)
        Xtr, Xte, var = pca.transform(Xtr_full), pca.transform(Xte_full), pca.explained_variance_ratio_.sum()
    else:
        Xtr, Xte, var = Xtr_full, Xte_full, 1.0
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(path, Xtr=Xtr, Xte=Xte, ytr=y[tr], yte=y[te],
                        Xtr_full=Xtr_full, Xte_full=Xte_full,
                        feature_names=np.array(feature_names),
                        groups_tr=groups[tr], groups_te=groups[te],
                        pca_var=var)
    print(f"[{name}] {len(X)} samples, {X.shape[1]} features -> {Xtr.shape[1]} network inputs "
          f"({var:.1%} variance kept)")
    print(f"[{name}] train {len(tr)} ({y[tr].mean():.1%} seizure, {len(set(groups[tr]))} recordings) | "
          f"test {len(te)} ({y[te].mean():.1%} seizure, {len(set(groups[te]))} recordings)")
    print(f"[{name}] saved {path}")


# =============================================================== 1. UCI
UCI_FS = 173.61            # Bonn sampling rate; each UCI chunk = 178 samples (~1 s)


def preprocess_uci(csv_path):
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]                                   # e.g. "X21.V1.791"
    feats = [c for c in df.columns if re.fullmatch(r"X\d+", c)]
    X = df[feats].values.astype(float)
    y = (df["y"].values == 1).astype(int)
    # ID = X<chunk>.<recording>; chunks of one recording share the recording part.
    # The class is part of the group so identically-named recordings from
    # different Bonn sets are never merged.
    rec = df[id_col].astype(str).str.split(".", n=1).str[1]
    groups = (df["y"].astype(str) + "_" + rec).values
    print(f"[uci] {len(df)} chunks from {len(set(groups))} recordings")
    # same 16 EEG features as CHB-MIT, computed on each 1-s single-channel chunk
    # (Bonn signals are already band-limited to 0.53-40 Hz, so no extra filter)
    F = np.vstack([window_features(x[None, :], UCI_FS) for x in X])
    standardise_and_save("uci", F, y, groups, FEATURE_NAMES)


# =============================================================== 2. CHB-MIT
def read_edf(path, wanted=CHANNELS):
    """Minimal EDF reader: returns (signals [n_channels, n_samples] in uV, fs)."""
    with open(path, "rb") as f:
        hdr = f.read(256)
        n_rec, rec_dur, ns = int(hdr[236:244]), float(hdr[244:252]), int(hdr[252:256])
        h = f.read(ns * 256)
        offs, o = {}, 0
        for key, size in [("label", 16), ("trans", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
                          ("dmin", 8), ("dmax", 8), ("pref", 80), ("nsamp", 8), ("res", 32)]:
            offs[key] = [h[o + i * size:o + (i + 1) * size].decode("latin-1").strip()
                         for i in range(ns)]
            o += size * ns
        labels = [l.upper() for l in offs["label"]]
        nsamp = np.array(offs["nsamp"], int)
        pmin, pmax = np.array(offs["pmin"], float), np.array(offs["pmax"], float)
        dmin, dmax = np.array(offs["dmin"], float), np.array(offs["dmax"], float)
        raw = np.fromfile(f, dtype="<i2")
    if n_rec < 0:
        n_rec = len(raw) // nsamp.sum()
    raw = raw[: n_rec * nsamp.sum()].reshape(n_rec, nsamp.sum())
    starts = np.concatenate([[0], np.cumsum(nsamp)])
    out = []
    for ch in wanted:
        if ch not in labels:
            return None, None                                # montage differs: skip file
        i = labels.index(ch)                                 # first occurrence (T8-P8 appears twice)
        d = raw[:, starts[i]:starts[i + 1]].ravel().astype(float)
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        out.append((d - dmin[i]) * gain + pmin[i])
    return np.vstack(out), nsamp[labels.index(wanted[0])] / rec_dur


def read_seizure_annotations(path):
    """Parse a PhysioNet MIT-format *.edf.seizures file -> list of (start_s, end_s)."""
    data = np.fromfile(path, dtype="<u2")
    t, i, fs, events = 0, 0, FS, []
    while i < len(data):
        w = int(data[i]); i += 1
        code, inc = w >> 10, w & 0x3FF
        if code == 0 and inc == 0:
            break                                            # end of file
        if code == 59:                                       # SKIP: 32-bit interval follows
            hi, lo = int(data[i]), int(data[i + 1]); i += 2
            val = (hi << 16) | lo
            t += val - (1 << 32) if val >= 1 << 31 else val
        elif code == 63:                                     # AUX string (padded to even length)
            n = (inc + 1) // 2
            txt = data[i:i + n].tobytes()[:inc].decode("latin-1", "ignore")
            m = re.search(r"time resolution:\s*(\d+)", txt)
            if m:
                fs = int(m.group(1))
            i += n
        elif code in (60, 61, 62):                           # NUM / SUB / CHN: ignore
            pass
        else:
            t += inc
            if code:
                events.append((code, t))
    starts = [s for c, s in events if c == 32]               # '[' seizure onset
    ends = [s for c, s in events if c == 33]                 # ']' seizure end
    return [(s / fs, e / fs) for s, e in zip(starts, ends)]


def read_summary(path):
    """Parse chbXX-summary.txt -> {file name: [(start_s, end_s), ...]}"""
    out, cur = {}, None
    starts = []
    for line in open(path, encoding="latin-1"):
        line = line.strip()
        if line.startswith("File Name:"):
            cur = line.split(":", 1)[1].strip()
            out[cur] = []
        elif cur and re.match(r"Seizure( \d+)? Start Time", line):
            starts.append(float(re.findall(r"(\d+)\s*seconds", line)[0]))
        elif cur and re.match(r"Seizure( \d+)? End Time", line):
            out[cur].append((starts.pop(0), float(re.findall(r"(\d+)\s*seconds", line)[0])))
    return out


def window_features(seg, fs):
    """seg: [channels, samples] -> 1-D feature vector (16 features per channel)."""
    d1 = np.diff(seg, axis=1)
    d2 = np.diff(d1, axis=1)
    var0, var1, var2 = seg.var(1) + 1e-12, d1.var(1) + 1e-12, d2.var(1) + 1e-12
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    f, P = welch(seg, fs=fs, nperseg=fs, axis=1)
    total = P[:, (f >= 0.5) & (f <= 40)].sum(1) + 1e-12
    rel = [P[:, (f >= lo) & (f < hi)].sum(1) / total for lo, hi in BANDS.values()]
    p = P[:, (f >= 0.5) & (f <= 40)] / total[:, None]
    spec_ent = -(p * np.log2(p + 1e-12)).sum(1)
    feats = [np.abs(seg).mean(1), seg.std(1), skew(seg, axis=1), kurtosis(seg, axis=1),
             np.abs(d1).sum(1) / seg.shape[1],                         # line length
             mobility, complexity,
             (np.diff(np.sign(seg - seg.mean(1, keepdims=True)), axis=1) != 0).mean(1),  # zero crossings
             np.ptp(seg, axis=1), np.log(total), *rel, spec_ent]
    return np.vstack(feats).T.ravel()                         # channel-major order


FEATURE_NAMES = ["mean_abs", "std", "skew", "kurtosis", "line_length", "hjorth_mobility",
                 "hjorth_complexity", "zero_cross", "peak_to_peak", "log_power",
                 *[f"rel_{b}" for b in BANDS], "spectral_entropy"]


def preprocess_chbmit(root, patients):
    rng = np.random.default_rng(SEED)
    sos = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
    win = WIN_SEC * FS
    X, y, groups = [], [], []
    for pat in patients:
        pdir = os.path.join(root, pat) if os.path.isdir(os.path.join(root, pat)) else root
        summ = glob.glob(os.path.join(pdir, f"{pat}-summary.txt"))
        summary = read_summary(summ[0]) if summ else {}
        files = sorted(glob.glob(os.path.join(pdir, f"{pat}_*.edf")))
        if not files:
            print(f"[chbmit] no EDF files for {pat} in {pdir}")
        for fp in files:
            fname = os.path.basename(fp)
            if os.path.exists(fp + ".seizures"):
                sz = read_seizure_annotations(fp + ".seizures")
            else:
                sz = summary.get(fname, [])               # no .seizures file = no seizures
            sig, fs = read_edf(fp)
            if sig is None or fs != FS:
                print(f"[chbmit] skip {fname} (montage or sampling rate differs)")
                continue
            sig = sosfiltfilt(sos, sig, axis=1)           # 0.5-40 Hz band-pass (also removes 60 Hz)
            n = sig.shape[1]
            in_sz = np.zeros(n, bool)
            for s, e in sz:
                in_sz[int(s * FS):int(e * FS)] = True
            sz_idx = [a for a in range(0, n - win + 1, SEIZURE_STEP_SEC * FS) if in_sz[a:a + win].all()]
            nm_idx = [a for a in range(0, n - win + 1, NORMAL_STEP_SEC * FS) if not in_sz[a:a + win].any()]
            for a in sz_idx:
                X.append(window_features(sig[:, a:a + win], FS)); y.append(1); groups.append(fname)
            # normal windows: store (file, offset) now, compute features after undersampling
            for a in nm_idx:
                X.append((fp, a)); y.append(0); groups.append(fname)
            print(f"[chbmit] {fname}: {len(sz)} seizure(s), {len(sz_idx)} seizure / {len(nm_idx)} normal windows")
    y, groups = np.array(y), np.array(groups)
    # undersample non-seizure windows to NORMAL_PER_SEIZURE x seizure windows,
    # spread evenly over files so seizure-free recordings are represented too
    n_keep = max(NORMAL_PER_SEIZURE * int(y.sum()), 1)
    nm = np.where(y == 0)[0]
    keep_nm = set(rng.choice(nm, size=min(n_keep, len(nm)), replace=False).tolist())
    # compute features for the kept normal windows (file by file, to load each EDF once)
    by_file = {}
    for i in keep_nm:
        by_file.setdefault(X[i][0], []).append(i)
    for fp, idx in by_file.items():
        sig, _ = read_edf(fp)
        sig = sosfiltfilt(sos, sig, axis=1)
        for i in idx:
            a = X[i][1]
            X[i] = window_features(sig[:, a:a + win], FS)
    keep = [i for i in range(len(y)) if y[i] == 1 or i in keep_nm]
    X = np.vstack([X[i] for i in keep])
    y, groups = y[keep], groups[keep]
    if y.sum() == 0:
        raise SystemExit("[chbmit] no seizure windows found - include files that contain seizures")
    names = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURE_NAMES]
    standardise_and_save("chbmit", X, y, groups, names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", help="path to Epileptic Seizure Recognition.csv")
    ap.add_argument("--chbmit", help="CHB-MIT root folder (containing chb01/, chb02/, ...)")
    ap.add_argument("--patients", nargs="+", default=["chb01", "chb02", "chb03", "chb05", "chb08"])
    a = ap.parse_args()
    if not (a.uci or a.chbmit):
        ap.error("give --uci and/or --chbmit")
    if a.uci:
        preprocess_uci(a.uci)
    if a.chbmit:
        preprocess_chbmit(a.chbmit, a.patients)
