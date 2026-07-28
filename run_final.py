"""Final grid: train every variant at its tuned configuration over N seeds, then
evaluate every method and aggregate the tables."""
import sys
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from paths import ROOT as _ROOT

GPU = ""
_GPU_LOCK = __import__("threading").Lock()
_GPU_N = 0


def _next_gpu():
    """Round-robin device assignment across the visible GPUs.

    Every spawned training would otherwise default to cuda:0 and leave the
    remaining devices idle.
    """
    global _GPU_N
    if GPU:
        return GPU
    try:
        import subprocess as _sp
        n = len(_sp.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        ).decode().strip().splitlines())
    except Exception:
        n = 1
    if n <= 1:
        return ""
    with _GPU_LOCK:
        g = _GPU_N % n
        _GPU_N += 1
    return str(g)


PY = sys.executable
CODE = f"{_ROOT}/code"
ROOT = f"{_ROOT}"

TRAIN_VARIANTS = ["proposed", "cnn", "siamese"]
ABLATIONS = ["no_metric", "no_temp", "fixed_proj"]
EVAL_METHODS = ["feature_svm", "cnn", "siamese", "cnn_biohash",
                "cnn_randproj", "cnn_compsens", "cnn_posthoc_ours", "proposed"]


def train_teacher(dataset, best, epochs, seed=0):
    """One unprotected encoder per dataset AND SEED, used as the teacher.

    Previously a single seed-0 teacher was reused for every seed.  Because
    make_split redraws the subject partition per seed, that teacher had been
    trained on subjects that are enrolment or test subjects under the other
    seeds, and its geometry was distilled into the protected encoder for those
    seeds.  The baselines are trained per seed, so the contamination was
    asymmetric in exactly the comparison the main table makes.
    """
    out = os.path.join(ROOT, "teachers")
    tag = f"{dataset}_cnn_L256_s{seed}"
    if os.path.exists(os.path.join(out, tag, "best.pt")):
        return os.path.join(out, tag)
    args = ([PY, os.path.join(CODE, "train.py"), "--dataset", dataset,
             "--variant", "cnn", "--seed", str(seed), "--epochs", str(epochs),
             "--out", out] + cfg_args(best, dataset, "cnn"))
    env = dict(os.environ, OMP_NUM_THREADS="8", TORCH_THREADS="8")
    g = _next_gpu()
    if g:
        env["CUDA_VISIBLE_DEVICES"] = g
    with open(os.path.join(ROOT, "logs", f"teacher_{dataset}_s{seed}.log"), "w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    p = os.path.join(out, tag)
    return p if os.path.exists(os.path.join(p, "best.pt")) else None


def cfg_args(best, dataset, variant):
    key = f"{dataset}/{variant}"
    if key not in best:
        # ablations inherit the tuned configuration of the full model
        key = f"{dataset}/proposed"
    out = []
    for k, v in best.get(key, {}).get("config", {}).items():
        out += [f"--{k}", str(v)]
    return out


def train_job(dataset, variant, seed, best, epochs, teachers=None, l4=0.0):
    # The template length comes from the tuned configuration, so the run
    # directory is not necessarily the L256 default; resolve it before deciding
    # whether the checkpoint already exists, otherwise finished runs are silently
    # retrained.
    L = dict(best.get(f"{dataset}/{variant}",
                      best.get(f"{dataset}/proposed", {})).get("config", {})
             ).get("L", 256)
    suffix = "" if l4 > 0 else "_nodistil"
    tag = f"{dataset}_{variant}_L{L}_s{seed}{suffix}"
    ck = os.path.join(ROOT, "runs", tag, "best.pt")
    if os.path.exists(ck):
        return f"  skip  {tag}"
    log = os.path.join(ROOT, "logs", f"train_{tag}.log")
    args = ([PY, os.path.join(CODE, "train.py"), "--dataset", dataset,
             "--variant", variant, "--seed", str(seed), "--epochs", str(epochs),
             "--tag", tag]
            + cfg_args(best, dataset, variant))
    # the protected variants additionally distil from the unprotected teacher
    t = (teachers or {}).get((dataset, seed))
    if t and l4 > 0 and variant in ("proposed", "no_metric", "no_temp", "fixed_proj"):
        args += ["--teacher", t, "--l4", str(l4)]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TORCH_THREADS="4")
    g = _next_gpu()
    if g:
        env["CUDA_VISIBLE_DEVICES"] = g
    with open(log, "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        return f"  FAIL  {tag}  (see {log})"
    try:
        with open(os.path.join(ROOT, "runs", tag, "history.json")) as f:
            h = json.load(f)
        return f"  ok    {tag}  val_eer={h['best_val_eer']:.3f}"
    except Exception:
        return f"  ok    {tag}"


def eval_job(dataset, method, seed):
    tag = f"{dataset}_{method}_L256_s{seed}"
    log = os.path.join(ROOT, "logs", f"eval_{tag}.log")
    args = [PY, os.path.join(CODE, "evaluate.py"), "--dataset", dataset,
            "--method", method, "--seed", str(seed)]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TORCH_THREADS="4")
    g = _next_gpu()
    if g:
        env["CUDA_VISIBLE_DEVICES"] = g
    with open(log, "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        return f"  FAIL  eval {tag} (see {log})"
    return "  " + open(log).read().strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--gpu", default="",
                    help="CUDA_VISIBLE_DEVICES for the spawned trainings")
    ap.add_argument("--with_ablations", type=int, default=1)
    ap.add_argument("--variants", default="",
                    help="comma-separated training variants; default is the full set")
    ap.add_argument("--only_teachers", type=int, default=0,
                    help="train the distillation teachers and stop")
    ap.add_argument("--skip_eval", type=int, default=0)
    ap.add_argument("--best", default=os.path.join(ROOT, "results/best_configs.json"))
    ap.add_argument("--l4", type=float, default=5.0,
                    help="distillation weight for the protected variants")
    a = ap.parse_args()

    global GPU
    GPU = a.gpu

    best = {}
    if os.path.exists(a.best):
        with open(a.best) as f:
            best = json.load(f)

    datasets = [d for d in a.datasets.split(",")
                if os.path.exists(os.path.join(ROOT, "data/proc", f"{d}.npz"))]
    variants = (a.variants.split(",") if a.variants
                else TRAIN_VARIANTS + (ABLATIONS if a.with_ablations else []))

    teachers = {}
    if a.l4 > 0:
        print("=== training distillation teachers ===", flush=True)
        for d in datasets:
            for sd in range(a.seeds):
                teachers[(d, sd)] = train_teacher(d, best, a.epochs, sd)
                print(f"  {d} seed {sd}: {teachers[(d, sd)]}", flush=True)

    if a.only_teachers:
        print("teachers only: done", flush=True)
        return

    jobs = [(d, v, s) for d in datasets for v in variants
            for s in range(a.seeds)]
    print(f"=== training {len(jobs)} runs ({a.par} in parallel) ===", flush=True)
    with ThreadPoolExecutor(max_workers=a.par) as ex:
        for msg in ex.map(
                lambda t: train_job(*t, best, a.epochs, teachers, a.l4), jobs):
            print(msg, flush=True)

    if a.skip_eval:
        print("skipping evaluation as requested", flush=True)
        return
    methods = EVAL_METHODS + (ABLATIONS if a.with_ablations else [])
    ejobs = [(d, m, s) for d in datasets for m in methods for s in range(a.seeds)]
    print(f"\n=== evaluating {len(ejobs)} runs ===", flush=True)
    with ThreadPoolExecutor(max_workers=max(4, a.par)) as ex:
        for msg in ex.map(lambda t: eval_job(*t), ejobs):
            print(msg, flush=True)

    print("\n=== aggregating ===", flush=True)
    subprocess.run([PY, os.path.join(CODE, "aggregate.py")])


if __name__ == "__main__":
    main()
