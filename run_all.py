"""
One command for the whole pipeline: preprocess -> train PSO/APO/SBO/QP-SBO -> figures.

Examples
  python run_all.py --uci "Epileptic Seizure Recognition.csv" --chbmit "D:/chb-mit"
  python run_all.py --chbmit "D:/chb-mit" --patients chb01 chb02 chb03 chb05 chb08
"""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--uci")
ap.add_argument("--chbmit")
ap.add_argument("--patients", nargs="+", default=["chb01", "chb02", "chb03", "chb05", "chb08"])
a = ap.parse_args()
if not (a.uci or a.chbmit):
    ap.error("give --uci and/or --chbmit")

py = sys.executable
run = lambda *cmd: subprocess.run([py, *cmd], check=True)

pre = ["preprocess.py"]
if a.uci:
    pre += ["--uci", a.uci]
if a.chbmit:
    pre += ["--chbmit", a.chbmit, "--patients", *a.patients]
run(*pre)

for name, given in [("uci", a.uci), ("chbmit", a.chbmit)]:
    if given:
        run("run_experiment.py", f"datasets/{name}.npz")
        run("make_figures.py", f"results_{name}.json")
print("\nDone. Results: results_*.json, summary_*.json  |  Figures: figures/")
