"""Identity leakage as a function of the number of observed revoked templates.

Reviewer comment R1.7 asks whether an adversary who accumulates several
superseded templates of the same subject gains anything.  We already answer that
for *linkage* (the linkability-vs-k curve in security.revocation), but not for
*identity recovery*, which is a different property: linkage asks whether two
templates belong to the same person, identity recovery asks whether the person
can be named.

Here the attacker is given the protected templates of every enrolled subject
under key versions 0..k-1, trains the same two-layer classifier used for the IRR
column of the security table, and is tested on query-side templates issued under
a fresh, unseen version k.  A flat curve means accumulating revoked templates
does not help; a rising curve would mean revocation leaks.

Writes results/tables/irr_vs_k.json.
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
from irr import fit_mlp
from models import make_transform, derive_seed
from security import _binarise, embeddings

from paths import ROOT as _ROOT

R = f"{_ROOT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--proc", default=f"{R}/data/proc")
    ap.add_argument("--runs", default=f"{R}/runs")
    ap.add_argument("--out", default=f"{R}/results/tables/irr_vs_k.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = defaultdict(list)

    for ds in a.datasets.split(","):
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        for seed in range(a.seeds):
            hits = sorted(glob.glob(os.path.join(
                a.runs, f"{ds}_proposed_L*_s{seed}", "best.pt")))
            if not hits:
                continue
            model, margs, _ = load_model(os.path.dirname(hits[0]), device)
            d, L, gamma = margs["d"], margs["L"], margs["gamma"]

            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)
            se = data["subject"][m["enrol"]]
            sq = data["subject"][m["test"]]
            Ze = embeddings(model, data["beats"][m["enrol"]], device)
            Zq = embeddings(model, data["beats"][m["test"]], device)

            # the fresh, never-observed version the attacker is tested against
            Rk, bk = make_transform(d, L, derive_seed("app-A", version=a.kmax, bind_user=False))
            Tq = _binarise(Zq, Rk, bk, gamma)

            for k in range(1, a.kmax + 1):
                # attacker's training material: k revoked versions, stacked
                Xs, ys = [], []
                for v in range(k):
                    Rv, bv = make_transform(d, L, derive_seed("app-A", version=v, bind_user=False))
                    Xs.append(_binarise(Ze, Rv, bv, gamma))
                    ys.append(se)
                X = np.concatenate(Xs).astype(np.float32)
                y = np.concatenate(ys)
                r = fit_mlp(X, y, np.asarray(Tq, np.float32), sq,
                            seed=seed, device=device)
                acc[f"{ds}/k{k}"].append(r)
                print(f"  {ds}/s{seed}/k={k}  IRR={r:.2f}%", flush=True)

    res = {}
    for k, v in acc.items():
        v = np.array([x for x in v if np.isfinite(x)], float)
        res[k] = dict(irr_mean=float(v.mean()),
                      irr_sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      n_seeds=int(v.size))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n=== IRR vs number of observed revoked templates ===")
    for k in sorted(res):
        print(f"  {k:20s} {res[k]['irr_mean']:6.2f} +- {res[k]['irr_sd']:.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
