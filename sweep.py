"""
Equal-budget hyper-parameter search.

Every variant is tuned with the same number of trials and selected on the same
criterion: verification EER on the validation subjects, which are disjoint from
both the training subjects and the evaluation subjects.  Tuning each system
rather than only the proposed one is what makes the later comparison a statement
about protected-space learning instead of about tuning effort (reviewer comment
R1.1).

Each trial is averaged over several seeds, because the validation pool holds only
a handful of identities and a single seed's EER is too noisy to rank
configurations.  Trials run concurrently, each in its own output directory.

The winning configuration per (dataset, variant) is written to best_configs.json
and consumed by the final runs.
"""
import sys
import argparse
import itertools
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

from paths import ROOT as _ROOT

STEPS_CAP = 0
GPU = ""
PY = sys.executable
CODE = f"{_ROOT}/code"

SPACE = {
    "lr": [1e-3, 3e-4],
    "wd": [1e-4, 1e-3],
    "l1": [0.5, 1.0, 2.0, 4.0],
    "l2": [0.01, 0.03, 0.1],
    "l3": [0.3, 1.0],
    "gamma": [2.0, 3.0, 5.0],
    "margin": [0.3, 0.5, 0.8],
    "metric_kind": ["triplet", "contrastive", "hamming"],
    "head": ["arc", "linear"],
    "arc_m": [0.2, 0.3, 0.4],
    "d": [512, 1024],
    "L": [256, 512],
    "drop": [0.1, 0.2],
}

USES = {
    "proposed":  ["lr", "wd", "l1", "l2", "l3", "gamma", "margin",
                  "metric_kind", "head", "arc_m", "d", "L", "drop"],
    "cnn":       ["lr", "wd", "head", "arc_m", "d", "drop"],
    "siamese":   ["lr", "wd", "l1", "margin", "metric_kind", "d", "drop"],
    "no_metric": ["lr", "wd", "l2", "l3", "gamma", "head", "arc_m", "d", "L", "drop"],
    "no_temp":   ["lr", "wd", "l1", "gamma", "margin", "metric_kind",
                  "head", "arc_m", "d", "L", "drop"],
    "fixed_proj": ["lr", "wd", "l1", "l2", "l3", "gamma", "margin",
                   "metric_kind", "head", "arc_m", "d", "L", "drop"],
}


def sample_configs(variant, n_trials, seed=0, fixed=None):
    """Random search over the variant's own sub-space.

    The full product is enormous for the protected variants, so configurations
    are drawn independently per key instead of materialising the grid.
    """
    fixed = fixed or {}
    keys = [k for k in USES[variant] if k not in fixed]
    rng = random.Random(seed)
    full = 1
    for k in keys:
        full *= len(SPACE[k])
    if full <= 4 * n_trials:                       # small space: shuffle it exactly
        grid = list(itertools.product(*[SPACE[k] for k in keys]))
        rng.shuffle(grid)
        return [dict(zip(keys, g), **fixed) for g in grid[:n_trials]]
    seen, out = set(), []
    while len(out) < n_trials and len(seen) < full:
        cfg = tuple(rng.choice(SPACE[k]) for k in keys)
        if cfg in seen:
            continue
        seen.add(cfg)
        out.append(dict(zip(keys, cfg), **fixed))
    return out


def run_one(dataset, variant, cfg, seed, epochs, out_root):
    args = [PY, os.path.join(CODE, "train.py"),
            "--dataset", dataset, "--variant", variant,
            "--seed", str(seed), "--epochs", str(epochs), "--out", out_root]
    if STEPS_CAP:
        args += ["--steps_per_epoch", str(STEPS_CAP)]
    for k, v in cfg.items():
        args += [f"--{k}", str(v)]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", TORCH_THREADS="2")
    if GPU:
        env["CUDA_VISIBLE_DEVICES"] = GPU
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    tag = f"{dataset}_{variant}_L{cfg.get('L', 256)}_s{seed}"
    hist = os.path.join(out_root, tag, "history.json")
    if r.returncode != 0 or not os.path.exists(hist):
        return None
    with open(hist) as f:
        return json.load(f).get("best_val_eer")


def run_trial(dataset, variant, cfg, idx, seeds, epochs, tmp):
    """One configuration, averaged over seeds, in its own directory."""
    out_root = os.path.join(tmp, f"{dataset}_{variant}_t{idx:03d}")
    vals = []
    for s in seeds:
        v = run_one(dataset, variant, cfg, s, epochs, out_root)
        if v is not None and v == v:
            vals.append(v)
    subprocess.run(["rm", "-rf", out_root])
    return (sum(vals) / len(vals)) if vals else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants", default="proposed,cnn,siamese")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--sweep_seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--gpu", default="",
                    help="CUDA_VISIBLE_DEVICES for the spawned trainings")
    ap.add_argument("--steps_per_epoch", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp", default=f"{_ROOT}/sweep_runs")
    ap.add_argument("--out", default=f"{_ROOT}/results/best_configs.json")
    ap.add_argument("--fix", default="",
                    help="pin hyper-parameters, e.g. metric_kind=hamming,L=512")
    ap.add_argument("--suffix", default="", help="label appended to the result key")
    a = ap.parse_args()

    global GPU
    GPU = a.gpu

    global STEPS_CAP
    STEPS_CAP = a.steps_per_epoch

    fixed = {}
    for kv in filter(None, a.fix.split(",")):
        k, v = kv.split("=", 1)
        fixed[k] = (float(v) if v.replace(".", "", 1).replace("e-", "", 1).isdigit()
                    else v)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    best = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            best = json.load(f)

    seeds = list(range(a.sweep_seeds))
    for variant in a.variants.split(","):
        cfgs = sample_configs(variant, a.trials, seed=a.seed, fixed=fixed)
        print(f"\n=== {a.dataset} / {variant}: {len(cfgs)} trials, "
              f"{a.par} concurrent, {a.sweep_seeds} seeds each ===", flush=True)
        with ThreadPoolExecutor(max_workers=a.par) as ex:
            scores = list(ex.map(
                lambda t: run_trial(a.dataset, variant, t[1], t[0], seeds,
                                    a.epochs, a.tmp),
                list(enumerate(cfgs))))
        rows = sorted(zip(scores, cfgs), key=lambda t: t[0])
        for sc, cfg in rows:
            print(f"  val_eer={sc:7.3f}  {cfg}", flush=True)
        best[f"{a.dataset}/{variant}{a.suffix}"] = dict(val_eer=rows[0][0], config=rows[0][1],
                                              n_trials=len(cfgs),
                                              sweep_seeds=a.sweep_seeds)
        print(f"  -> BEST val_eer={rows[0][0]:.3f}  {rows[0][1]}", flush=True)
        with open(a.out, "w") as f:
            json.dump(best, f, indent=2)

    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
