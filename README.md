# Epileptic Seizure Detection with Quantum Parity SBO

A neural network (input → 10 tanh → 1 sigmoid) is trained by four metaheuristics
and compared on two EEG datasets:

| Optimizer | Description |
|---|---|
| PSO | Particle Swarm Optimization |
| APO | Artificial Protozoa Optimizer |
| SBO | Satin Bowerbird Optimizer |
| **QP-SBO** | **Quantum Parity SBO (proposed)**: qubit-guided moves + rotation gate + parity operator |

All four get the same budget (15,000 fitness evaluations, population 30) and
10 independent runs.

## 1. Install

```
pip install numpy scipy pandas scikit-learn matplotlib
```

No MNE or pyEDFlib is needed. EDF files and `.edf.seizures` annotations are read directly.

## 2. Put the data in place

```
Epileptic Seizure Recognition.csv
chb-mit/
  chb01/  chb01_01.edf ... chb01_03.edf  chb01_03.edf.seizures ... chb01-summary.txt
  chb02/  ...
```

Seizure times come from the `.edf.seizures` files. If those are missing, they come from `chbXX-summary.txt`.
A file with neither is treated as seizure-free, which is correct for CHB-MIT.

## 3. Run everything

```
python run_all.py --uci "Epileptic Seizure Recognition.csv" --chbmit chb-mit --patients chb01 chb02 chb03 chb05 chb08
```

Or run the steps one at a time:

```
python preprocess.py --uci "Epileptic Seizure Recognition.csv" --chbmit chb-mit --patients chb01 chb02 chb03 chb05 chb08
python run_experiment.py datasets/uci.npz
python run_experiment.py datasets/chbmit.npz
python make_figures.py results_uci.json
python make_figures.py results_chbmit.json
```

Outputs:
- `results_<dataset>.json`: every run, every metric and every convergence curve.
- `summary_<dataset>.json`: mean ± sd table and Wilcoxon p-values.
- `figures/<dataset>_fig1..fig5.png`.

## What preprocessing does

| Step | UCI (Bonn) | CHB-MIT |
|---|---|---|
| Signal | 11,500 one-second single-channel chunks (178 samples, 173.61 Hz) | 18 bipolar scalp channels, 256 Hz |
| Filtering | none (already 0.53–40 Hz) | 4th-order Butterworth band-pass 0.5–40 Hz, zero-phase |
| Segmentation | given (1 s) | 4-s windows; 1-s step inside seizures, 4-s step elsewhere; windows straddling a seizure boundary are dropped |
| Labels | class 1 = seizure, classes 2–5 = non-seizure | inside annotated seizure = 1, else 0 |
| Balancing | natural (20 % seizure) | non-seizure windows undersampled to 4 per seizure window (20 % seizure) |
| Features (per channel) | 16 | 16 × 18 channels = 288 |
| Split | by recording (500 Bonn recordings) | by EDF file |
| Standardisation | z-score, fitted on train only | z-score, fitted on train only |
| Reduction | none (16 inputs) | PCA → 20 inputs, fitted on train only |

The 16 features are:
- **Time domain:** mean |x|, standard deviation, skewness, kurtosis, line length, Hjorth mobility, Hjorth complexity, zero-crossing rate, peak-to-peak and log total power.
- **Relative band power:** delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) and gamma (30–40 Hz).
- **Spectral entropy.**

The split is by recording, so segments of one recording never appear in both
train and test. This avoids the leakage that inflates many published UCI results.
