"""
Identity recovery rate (IRR).

The auxiliary adversary is a two-layer MLP that receives a protected template and
predicts the subject identity.  It is trained on the enrolment-side templates of
the evaluation subjects and tested on their query-side templates, so training and
testing use disjoint *recordings* of the same enrolled identities.  This is the
closed-set identity-recovery task of the manuscript: it measures how much
identity information survives in the stored template, and it is deliberately
distinct from the attribute-leakage experiment and from unlinkability.

IRR bears on property P2 (resistance to recovering a discriminative
representation).  A low value does not establish non-invertibility: it bounds
what this particular adversary extracts, not what a stronger one could.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import protocols
from evaluate import load_model, templates
from models import (BioHashing, CompressiveSensing, PostHocCancelable,
                    RandomProjection)
from security import _binarise, embeddings
from models import derive_seed, make_transform

from paths import ROOT as _ROOT

POSTHOC = {"cnn_biohash": BioHashing, "cnn_randproj": RandomProjection,
           "cnn_compsens": CompressiveSensing,
           "cnn_posthoc_ours": PostHocCancelable}


def fit_mlp(Xtr, ytr, Xte, yte, seed=0, hidden=256, epochs=200, device="cpu"):
    """Two-layer MLP identity classifier; returns top-1 recovery rate (%)."""
    classes = np.unique(ytr)
    remap = {int(c): i for i, c in enumerate(classes)}
    ytr_i = np.array([remap[int(v)] for v in ytr], np.int64)
    keep = np.array([int(v) in remap for v in yte])
    if keep.sum() == 0:
        return float("nan")
    yte_i = np.array([remap[int(v)] for v in yte[keep]], np.int64)
    Xte = Xte[keep]

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, len(classes))).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr_i, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, dtype=torch.float32,
                                device=device)).argmax(1).cpu().numpy()
    return 100.0 * float((pred == yte_i).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--methods",
                    default="cnn,cnn_biohash,cnn_randproj,cnn_compsens,proposed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", default=f"{_ROOT}/runs")
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--out", default=f"{_ROOT}/results/tables/irr.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = defaultdict(list)

    for ds in a.datasets.split(","):
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        for seed in range(a.seeds):
            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)
            se, sq = data["subject"][m["enrol"]], data["subject"][m["test"]]

            for meth in a.methods.split(","):
                backbone = "cnn" if meth in POSTHOC else meth
                hits = sorted(glob.glob(os.path.join(
                    a.runs, f"{ds}_{backbone}_L*_s{seed}", "best.pt")))
                if not hits:
                    continue
                model, margs, _ = load_model(os.path.dirname(hits[0]), device)
                d = margs["d"]

                if meth == "proposed":
                    L, gamma = margs["L"], margs["gamma"]
                    R, b = make_transform(d, L, derive_seed("app-A", version=0, bind_user=False))
                    Te = _binarise(embeddings(model, data["beats"][m["enrol"]],
                                              device), R, b, gamma)
                    Tq = _binarise(embeddings(model, data["beats"][m["test"]],
                                              device), R, b, gamma)
                elif meth in POSTHOC:
                    tf = POSTHOC[meth](d=d, L=256, key="app-A")
                    Te = tf(templates(model, data["beats"][m["enrol"]], device,
                                      hard=False))
                    Tq = tf(templates(model, data["beats"][m["test"]], device,
                                      hard=False))
                else:                                  # unprotected embedding
                    Te = embeddings(model, data["beats"][m["enrol"]], device)
                    Tq = embeddings(model, data["beats"][m["test"]], device)

                r = fit_mlp(np.asarray(Te, np.float32), se,
                            np.asarray(Tq, np.float32), sq,
                            seed=seed, device=device)
                acc[f"{ds}/{meth}"].append(r)
                print(f"  {ds}/{meth}/s{seed}  IRR={r:.2f}%", flush=True)

    res = {}
    for k, v in acc.items():
        v = np.array([x for x in v if np.isfinite(x)], float)
        res[k] = dict(irr_mean=float(v.mean()),
                      irr_sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      n_seeds=int(v.size))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== IRR (closed-set identity recovery from protected templates) ===")
    for k in sorted(res):
        print(f"  {k:28s} {res[k]['irr_mean']:6.2f} +- {res[k]['irr_sd']:.2f}"
              f"  (n={res[k]['n_seeds']})")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
