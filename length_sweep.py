"""
Controlled template-length sweep with inference timing.

The template length is the only quantity varied: encoder, losses, optimiser,
schedule, augmentation and seeds are held at the tuned configuration of the
dataset, so the resulting curve is attributable to L alone.  For each length the
script reports verification EER, cross-application linkability, the
stored-template guessing attack, and the per-sample inference cost of producing
one protected template.

Inference time is measured end to end on the deployment path -- encoder forward,
key-derived projection, and binarisation -- after warm-up, with CUDA
synchronisation, and is reported per heartbeat.
"""
import sys
import argparse
import glob
import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch

import protocols
from evaluate import load_model, templates
from metrics import (attack_success_rate, eer_from_scores, hamming_similarity,
                     linkability_auc, linkability_dsys, tpr_at_far)
from models import derive_seed, make_transform
from security import _binarise, embeddings, per_identity

from paths import ROOT as _ROOT

GPU = ""
PY = sys.executable
CODE = f"{_ROOT}/code"
ROOT = f"{_ROOT}"


def train_one(dataset, L, seed, best, epochs, out_root):
    tag = f"{dataset}_proposed_L{L}_s{seed}"
    if os.path.exists(os.path.join(out_root, tag, "best.pt")):
        return os.path.join(out_root, tag)
    cfg = dict(best[f"{dataset}/proposed"]["config"])
    cfg["L"] = L
    args = [PY, os.path.join(CODE, "train.py"), "--dataset", dataset,
            "--variant", "proposed", "--seed", str(seed),
            "--epochs", str(epochs), "--out", out_root]
    for k, v in cfg.items():
        args += [f"--{k}", str(v)]
    t = os.path.join(ROOT, "teachers", f"{dataset}_cnn_L256_s0")
    if os.path.exists(os.path.join(t, "best.pt")):
        args += ["--teacher", t, "--l4", "5.0"]
    env = dict(os.environ, OMP_NUM_THREADS="4", TORCH_THREADS="4")
    if GPU:
        env["CUDA_VISIBLE_DEVICES"] = GPU
    with open(os.path.join(ROOT, "logs", f"lsweep_{tag}.log"), "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    p = os.path.join(out_root, tag)
    return p if r.returncode == 0 and os.path.exists(
        os.path.join(p, "best.pt")) else None


@torch.no_grad()
def time_per_sample(model, X, device, repeats=20, batch=256):
    """Milliseconds to turn one heartbeat into a stored protected template."""
    xb = torch.from_numpy(X[:batch]).float().to(device)
    for _ in range(5):                                   # warm-up
        model.template(xb, hard=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.template(xb, hard=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    return 1000.0 * dt / len(xb)


def evaluate_L(run_dir, data, m, device, seed, n_attempts=200):
    model, margs, _ = load_model(run_dir, device)
    d, L, gamma = margs["d"], margs["L"], margs["gamma"]
    se, sq = data["subject"][m["enrol"]], data["subject"][m["test"]]

    Te = templates(model, data["beats"][m["enrol"]], device, hard=True).astype(np.uint8)
    Tq = templates(model, data["beats"][m["test"]], device, hard=True).astype(np.uint8)
    sc, lb, _, _ = protocols.aggregate_scores(hamming_similarity(Tq, Te), se, sq)
    eer = eer_from_scores(sc, lb)[0]

    # threshold from the validation subjects, exactly as in the main protocol
    Ve = templates(model, data["beats"][m["val_enrol"]], device, hard=True).astype(np.uint8)
    Vq = templates(model, data["beats"][m["val_query"]], device, hard=True).astype(np.uint8)
    vsc, vlb, _, _ = protocols.aggregate_scores(
        hamming_similarity(Vq, Ve), data["subject"][m["val_enrol"]],
        data["subject"][m["val_query"]])
    thr = float(tpr_at_far(vsc, vlb, 0.01)[1])

    Ze = embeddings(model, data["beats"][m["enrol"]], device)
    Zq = embeddings(model, data["beats"][m["test"]], device)
    Ra, ba = make_transform(d, L, derive_seed("app-A", version=0, bind_user=False))
    Rb, bb = make_transform(d, L, derive_seed("app-B", version=0, bind_user=False))
    Zall = np.concatenate([Ze, Zq]); sall = np.concatenate([se, sq])
    Pa, ids = per_identity(_binarise(Zall, Ra, ba, gamma), sall)
    Pb, _ = per_identity(_binarise(Zall, Rb, bb, gamma), sall)
    S = hamming_similarity(Pa, Pb)
    n = len(ids)
    link = linkability_auc(np.diag(S), S[~np.eye(n, dtype=bool)])
    dsys = linkability_dsys(np.diag(S), S[~np.eye(n, dtype=bool)])

    Pe, _ = per_identity(_binarise(Ze, Ra, ba, gamma), se)
    rng = np.random.default_rng(seed)
    p = Pe.mean(0)
    hit = []
    for i in range(len(Pe)):
        cand = (rng.random((n_attempts, Pe.shape[1])) < p).astype(np.uint8)
        hit.append(float((hamming_similarity(cand, Pe[i:i + 1]).ravel() >= thr).any()))
    asr = 100.0 * float(np.mean(hit))

    ms = time_per_sample(model, data["beats"][m["test"]], device)
    return dict(eer=eer, link_auc=link, link_dsys=dsys, asr=asr, ms=ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ecgid")
    ap.add_argument("--lengths", default="64,128,256,512")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--gpu", default="",
                    help="CUDA_VISIBLE_DEVICES for the spawned trainings")
    ap.add_argument("--runs", default=f"{_ROOT}/lsweep_runs")
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--best", default=f"{_ROOT}/results/best_configs.json")
    ap.add_argument("--out", default=f"{_ROOT}/results/tables/template_length.json")
    a = ap.parse_args()

    global GPU
    GPU = a.gpu

    with open(a.best) as f:
        best = json.load(f)
    lengths = [int(x) for x in a.lengths.split(",")]
    os.makedirs(a.runs, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    jobs = [(L, s) for L in lengths for s in range(a.seeds)]
    print(f"=== training {len(jobs)} runs ({a.par} concurrent) ===", flush=True)
    with ThreadPoolExecutor(max_workers=a.par) as ex:
        list(ex.map(lambda t: train_one(a.dataset, t[0], t[1], best,
                                        a.epochs, a.runs), jobs))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = protocols.load(os.path.join(a.proc, f"{a.dataset}.npz"))
    acc = defaultdict(lambda: defaultdict(list))
    for L in lengths:
        for seed in range(a.seeds):
            rd = os.path.join(a.runs, f"{a.dataset}_proposed_L{L}_s{seed}")
            if not os.path.exists(os.path.join(rd, "best.pt")):
                continue
            split = protocols.make_split(data, a.dataset, seed=seed)
            m = protocols.masks(data, split)
            r = evaluate_L(rd, data, m, device, seed)
            for k, v in r.items():
                acc[L][k].append(v)
            print(f"  L={L:4d} s{seed}  EER={r['eer']:.2f}  link={r['link_auc']:.3f}"
                  f"  ASR={r['asr']:.1f}  {r['ms']:.3f} ms", flush=True)

    res = {}
    for L in lengths:
        if L not in acc:
            continue
        res[str(L)] = {k: dict(mean=float(np.mean(v)),
                               sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                       for k, v in acc[L].items()}
        res[str(L)]["n_seeds"] = len(acc[L]["eer"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== template length sweep ===")
    print(f"{'L':>6s}{'EER(%)':>12s}{'Link.AUC':>11s}{'ASR(%)':>10s}{'Time(ms)':>11s}")
    for L in lengths:
        if str(L) not in res:
            continue
        r = res[str(L)]
        print(f"{L:6d}{r['eer']['mean']:8.2f}±{r['eer']['sd']:.2f}"
              f"{r['link_auc']['mean']:11.3f}{r['asr']['mean']:10.1f}"
              f"{r['ms']['mean']:11.3f}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
