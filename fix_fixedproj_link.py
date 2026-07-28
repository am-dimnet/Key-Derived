"""
Correct linkability measurement for the fixed-projection ablation.

That variant is trained with a projection that does not depend on the
application key, so the two applications share one transform and the same
subject produces the *same* protected template in both.  The earlier sweep
re-derived a key-dependent transform for every variant, which silently gave the
fixed-projection variant the protection it is supposed to lack.  Here both
applications use the variant's own transform, which is what the ablation is
meant to test (reviewer comment R2.3).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import protocols
from evaluate import load_model
from metrics import hamming_similarity, linkability_auc, linkability_dsys
from models import derive_seed, make_transform
from security import _binarise, embeddings, per_identity

from paths import ROOT as _ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
out = defaultdict(lambda: defaultdict(list))

for ds in ("ecgid", "heartprint", "ptbdb"):
    data = protocols.load(f"{_ROOT}/data/proc/{ds}.npz")
    for seed in range(5):
        split = protocols.make_split(data, ds, seed=seed)
        m = protocols.masks(data, split)
        for var in ("proposed", "fixed_proj"):
            hv = sorted(glob.glob(
                f"{_ROOT}/runs/{ds}_{var}_L*_s{seed}/best.pt"))
            if not hv:
                continue
            model, margs, _ = load_model(os.path.dirname(hv[0]), device)
            d, L, gamma = margs["d"], margs["L"], margs["gamma"]
            Z = np.concatenate([
                embeddings(model, data["beats"][m["enrol"]], device),
                embeddings(model, data["beats"][m["test"]], device)])
            subj = np.concatenate([data["subject"][m["enrol"]],
                                   data["subject"][m["test"]]])
            if var == "fixed_proj":
                # one transform, shared by both applications
                R, b = make_transform(d, L, derive_seed("FIXED-PROJECTION",
                                                        version=0))
                Ra, ba = Rb, bb = R, b
            else:
                Ra, ba = make_transform(d, L, derive_seed("app-A", version=0, bind_user=False))
                Rb, bb = make_transform(d, L, derive_seed("app-B", version=0, bind_user=False))
            Pa, ids = per_identity(_binarise(Z, Ra, ba, gamma), subj)
            Pb, _ = per_identity(_binarise(Z, Rb, bb, gamma), subj)
            S = hamming_similarity(Pa, Pb)
            n = len(ids)
            mated, nonmated = np.diag(S), S[~np.eye(n, dtype=bool)]
            k = f"{ds}/{var}"
            out[k]["link_auc"].append(linkability_auc(mated, nonmated))
            out[k]["link_dsys"].append(linkability_dsys(mated, nonmated))

res = {k: {kk: float(np.nanmean(vv)) for kk, vv in v.items()}
       for k, v in out.items()}
with open(f"{_ROOT}/results/tables/fixedproj_link.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"{'key':26s}{'linkAUC':>9s}{'Dsys':>8s}")
for k in sorted(res):
    print(f"{k:26s}{res[k]['link_auc']:9.3f}{res[k]['link_dsys']:8.3f}")
