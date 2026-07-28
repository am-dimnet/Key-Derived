"""Security, revocability and identity-recovery measurements for the
single-variable control (R1.1).

The control -- our key-derived transform applied post hoc to the frozen CNN
encoder -- was previously carried only through the accuracy tables, so every
security comparison in the paper was made against BioHashing, random projection
or compressive sensing, i.e. against a *different transform*.  That is the exact
confound the reviewer asked us to remove.  This script measures the control with
the same transform and template length as the proposed system, so the only
remaining difference is whether the encoder was trained inside the protected
space.

It also re-measures the control at the template length the proposed system
actually uses on each dataset, which is L=512 on ECG-ID.  The evaluation driver
resolves L from the backbone checkpoint directory, and the CNN backbone lives at
L=256, so the ECG-ID control had silently been measured at L=256 while the
proposed system ran at L=512.

Writes results/tables/control_security.json.
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
from models import PostHocCancelable, make_transform, derive_seed
from security import (attack_success_rate, hamming_similarity, linkability_auc,
                      linkability_dsys, per_identity)

from paths import ROOT as _ROOT

R = f"{_ROOT}"


def val_threshold_binary(model, data, m, tf, far=0.01):
    """FAR=1% operating point on the validation subjects, never re-tuned."""
    device = next(model.parameters()).device
    Ze = templates(model, data["beats"][m["val_enrol"]], device, hard=False)
    Zq = templates(model, data["beats"][m["val_query"]], device, hard=False)
    Pe, ids = per_identity(tf(Ze), data["subject"][m["val_enrol"]])
    S = hamming_similarity(tf(Zq), Pe)
    sc, lb, _, _ = protocols.aggregate_scores(
        S, ids, data["subject"][m["val_query"]])
    imp = np.sort(sc[lb == 0])
    return float(imp[int(np.ceil((1 - far) * len(imp))) - 1]) if len(imp) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--proc", default=f"{R}/data/proc")
    ap.add_argument("--runs", default=f"{R}/runs")
    ap.add_argument("--best", default=f"{R}/results/best_configs.json")
    ap.add_argument("--out", default=f"{R}/results/tables/control_security.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = json.load(open(a.best)) if os.path.exists(a.best) else {}
    acc = defaultdict(lambda: defaultdict(list))

    for ds in a.datasets.split(","):
        # the control must use the template length of the PROPOSED system on
        # this dataset, not the length of the backbone checkpoint directory
        L = int(best.get(f"{ds}/proposed", {}).get("config", {}).get("L", 256))
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        print(f"=== {ds}  (control at L={L}) ===", flush=True)

        for seed in range(a.seeds):
            hits = sorted(glob.glob(os.path.join(
                a.runs, f"{ds}_cnn_L*_s{seed}", "best.pt")))
            if not hits:
                print(f"  s{seed}: no cnn checkpoint", flush=True)
                continue
            model, margs, _ = load_model(os.path.dirname(hits[0]), device)
            d = margs["d"]

            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)
            se = data["subject"][m["enrol"]]
            sq = data["subject"][m["test"]]

            Ze = templates(model, data["beats"][m["enrol"]], device, hard=False)
            Zq = templates(model, data["beats"][m["test"]], device, hard=False)

            tf = PostHocCancelable(d=d, L=L, key="app-A")
            thr = val_threshold_binary(model, data, m, tf)

            # --- verification accuracy at this L
            Pe, ids = per_identity(tf(Ze), se)
            S = hamming_similarity(tf(Zq), Pe)
            sc, lb, _, _ = protocols.aggregate_scores(S, ids, sq)
            fpr = np.sort(sc[lb == 0])[::-1]
            # EER by scanning thresholds over the pooled scores
            order = np.argsort(-sc)
            s_sorted, l_sorted = sc[order], lb[order]
            P, N = l_sorted.sum(), len(l_sorted) - l_sorted.sum()
            tp = np.cumsum(l_sorted)
            fp = np.cumsum(1 - l_sorted)
            far = fp / max(N, 1)
            frr = 1 - tp / max(P, 1)
            i = int(np.argmin(np.abs(far - frr)))
            eer = float((far[i] + frr[i]) / 2 * 100)

            # --- cross-application unlinkability (P3)
            Zall = np.concatenate([Ze, Zq])
            sall = np.concatenate([se, sq])
            Pa, ids_a = per_identity(PostHocCancelable(d, L, key="app-A")(Zall), sall)
            Pb, _ = per_identity(PostHocCancelable(d, L, key="app-B")(Zall), sall)
            Sl = hamming_similarity(Pa, Pb)
            n = len(ids_a)
            mated, nonmated = np.diag(Sl), Sl[~np.eye(n, dtype=bool)]

            # --- template-only guessing attack (P4), same budget as elsewhere
            rng = np.random.default_rng(seed)
            p = Pe.mean(0)
            hit = 0
            for _ in range(200):
                g = (rng.random(L) < p).astype(np.uint8)
                if hamming_similarity(g[None, :], Pe).max() >= thr:
                    hit += 1
            t_only = 100.0 * hit / 200

            # --- revocation with utility (RevScore) at a new key version
            tf2 = PostHocCancelable(d=d, L=L, key="app-A", version=1)
            Pe2, ids2 = per_identity(tf2(Ze), se)
            S2 = hamming_similarity(tf2(Zq), Pe2)
            sc2, lb2, _, _ = protocols.aggregate_scores(S2, ids2, sq)
            o2 = np.argsort(-sc2)
            l2 = lb2[o2]
            P2, N2 = l2.sum(), len(l2) - l2.sum()
            far2 = np.cumsum(1 - l2) / max(N2, 1)
            frr2 = 1 - np.cumsum(l2) / max(P2, 1)
            j = int(np.argmin(np.abs(far2 - frr2)))
            eer_new = float((far2[j] + frr2[j]) / 2 * 100)

            # --- superseded template acceptance
            S_stale = hamming_similarity(tf(Zq), Pe2)
            sc_s, lb_s, _, _ = protocols.aggregate_scores(S_stale, ids2, sq)
            stale = attack_success_rate(sc_s[lb_s == 1], thr)

            k = f"{ds}/cnn_posthoc_ours"
            acc[k]["L"].append(L)
            acc[k]["eer"].append(eer)
            acc[k]["eer_new"].append(eer_new)
            # RevScore must be Eq. (2), 1 - |new-old|/old, computed per seed
            # and then averaged -- the definition every other row of the
            # revocability table uses (fill_tables.py, extra_tables.py).  An
            # EER ratio clamped at 1.0 is a different quantity and would make
            # this row incomparable with the rest of the column.
            acc[k]["revscore"].append(1.0 - abs(eer_new - eer) / eer)
            acc[k]["link_auc"].append(linkability_auc(mated, nonmated))
            acc[k]["link_dsys"].append(linkability_dsys(mated, nonmated))
            acc[k]["tmpl_only"].append(t_only)
            acc[k]["stale_acceptance_pct"].append(stale)
            print(f"  s{seed}: EER={eer:.2f} newEER={eer_new:.2f} "
                  f"link={acc[k]['link_auc'][-1]:.3f} ASR={t_only:.1f} "
                  f"stale={stale:.1f}", flush=True)

    res = {}
    for k, v in acc.items():
        res[k] = {kk: (float(np.mean(vv)) if kk != "L" else int(vv[0]))
                  for kk, vv in v.items()}
        res[k]["eer_sd"] = float(np.std(v["eer"], ddof=1)) if len(v["eer"]) > 1 else 0.0
        res[k]["n_seeds"] = len(v["eer"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n=== control security ===")
    for k in sorted(res):
        r = res[k]
        print(f"  {k:30s} L={r['L']} EER={r['eer']:.2f} new={r['eer_new']:.2f} "
              f"rev={r['revscore']:.3f} link={r['link_auc']:.3f} "
              f"ASR={r['tmpl_only']:.1f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
