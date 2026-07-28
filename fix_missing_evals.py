"""Re-run only the evaluations whose result file is absent.

The full-grid driver aborts a job on CUDA OOM, so a handful of the largest
(PTB) evaluations can be left behind.  This fills the gaps at low concurrency
and with the devices assigned round-robin.
"""
import sys
import glob
import itertools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from paths import ROOT as _ROOT

R = f"{_ROOT}"
DATASETS = ("ecgid", "heartprint", "ptbdb")
METHODS = ("feature_svm", "cnn", "siamese", "cnn_biohash", "cnn_randproj",
           "cnn_compsens", "cnn_posthoc_ours", "proposed",
           "no_metric", "no_temp", "fixed_proj")
SEEDS = range(5)

missing = [(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS
           if not glob.glob(f"{R}/results/result_{d}_{m}_L*_s{s}.json")]
print(f"missing evaluations: {len(missing)}")
for j in missing:
    print("   ", j)

_gpu = itertools.cycle(("0", "1"))


def run(t):
    d, m, s = t
    env = dict(os.environ, OMP_NUM_THREADS="4", TORCH_THREADS="4",
               CUDA_VISIBLE_DEVICES=next(_gpu))
    log = f"{R}/logs/fix_eval_{d}_{m}_s{s}.log"
    with open(log, "w") as f:
        r = subprocess.run(
            [sys.executable, f"{R}/code/evaluate.py",
             "--dataset", d, "--method", m, "--seed", str(s)],
            stdout=f, stderr=subprocess.STDOUT, env=env)
    return ("ok   " if r.returncode == 0 else "FAIL ") + f"{d}/{m}/s{s}"


if missing:
    with ThreadPoolExecutor(max_workers=4) as ex:
        for msg in ex.map(run, missing):
            print("  ", msg, flush=True)

print("results now:", len(glob.glob(f"{R}/results/result_*.json")))
