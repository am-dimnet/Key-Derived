"""
Two evaluation-only quantities the manuscript tables still need.

  RevScore      EER before and after the application key is replaced, i.e. how
                much verification utility survives re-issuance (Section IV-C).
                This is a utility measure, not a security one (R1.7).
  cross-session same-session vs cross-session EER, reported the same way for
                Heartprint and for ECG-ID, whose enrol/test split is also a
                cross-session setup (R2.4).
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
from metrics import eer_from_scores, hamming_similarity, cosine_similarity

from paths import ROOT as _ROOT


def eer_for_key(model, data, m, device, app_key, version):
    """Re-derive the transform, re-enrol, and re-measure EER."""
    if model.cancel is not None:
        model.cancel.rekey(app_key, version=version)
    Xe, Xq = data["beats"][m["enrol"]], data["beats"][m["test"]]
    se, sq = data["subject"][m["enrol"]], data["subject"][m["test"]]
    prot = model.cancel is not None
    Te = templates(model, Xe, device, hard=prot)
    Tq = templates(model, Xq, device, hard=prot)
    S = (hamming_similarity(Tq.astype(np.uint8), Te.astype(np.uint8))
         if prot else cosine_similarity(Tq, Te))
    sc, lb, _, _ = protocols.aggregate_scores(S, se, sq)
    return eer_from_scores(sc, lb)[0]


def session_split_eer(model, data, m, split, device, dataset):
    """Same-session and cross-session EER within the evaluation subjects.

    Same-session pairs enrol and query records drawn from the same acquisition
    session; cross-session uses the protocol's own enrol/test split.  ECG-ID has
    no session labels, so consecutive recordings of one subject are treated as
    the session axis, which is what its enrol/test protocol already does.
    """
    prot = model.cancel is not None
    dev = device

    def score(mask_e, mask_q):
        Xe, Xq = data["beats"][mask_e], data["beats"][mask_q]
        se, sq = data["subject"][mask_e], data["subject"][mask_q]
        if len(Xe) == 0 or len(Xq) == 0:
            return float("nan")
        Te = templates(model, Xe, dev, hard=prot)
        Tq = templates(model, Xq, dev, hard=prot)
        S = (hamming_similarity(Tq.astype(np.uint8), Te.astype(np.uint8))
             if prot else cosine_similarity(Tq, Te))
        sc, lb, _, _ = protocols.aggregate_scores(S, se, sq)
        return eer_from_scores(sc, lb)[0]

    cross = score(m["enrol"], m["test"])

    # same-session: split the enrolment records of each subject in half
    rec = data["record"]
    enrol_recs = np.asarray(split["enrol_records"])
    by_subj = defaultdict(list)
    for r in enrol_recs:
        s = int(data["subject"][rec == r][0])
        by_subj[s].append(int(r))
    e_half, q_half = [], []
    for s, rs in by_subj.items():
        rs = sorted(rs)
        if len(rs) >= 2:
            e_half += rs[: len(rs) // 2]
            q_half += rs[len(rs) // 2:]
    if e_half and q_half:
        same = score(np.isin(rec, e_half), np.isin(rec, q_half))
    else:
        # only one enrolment record per subject: split its beats in two halves,
        # which stays within a session but no longer across recordings
        same = float("nan")
        mask = m["enrol"]
        idx = np.flatnonzero(mask)
        if len(idx) > 20:
            half = len(idx) // 2
            a = np.zeros(len(mask), bool); a[idx[:half]] = True
            b = np.zeros(len(mask), bool); b[idx[half:]] = True
            same = score(a, b)
    return same, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--methods", default="proposed,cnn,siamese,cnn_biohash")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", default=f"{_ROOT}/runs")
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--out", default=f"{_ROOT}/results/tables")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rev, sess = defaultdict(list), defaultdict(list)

    for ds in a.datasets.split(","):
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        for meth in a.methods.split(","):
            backbone = {"cnn_biohash": "cnn"}.get(meth, meth)
            for seed in range(a.seeds):
                hits = sorted(glob.glob(os.path.join(
                    a.runs, f"{ds}_{backbone}_L*_s{seed}", "best.pt")))
                if not hits:
                    continue
                run_dir = os.path.dirname(hits[0])
                split = protocols.make_split(data, ds, seed=seed)
                m = protocols.masks(data, split)
                model, margs, _ = load_model(run_dir, device)

                if model.cancel is not None:
                    old = eer_for_key(model, data, m, device, "app-A", 0)
                    new = eer_for_key(model, data, m, device, "app-A", 1)
                    if np.isfinite(old) and old > 0:
                        rev[f"{ds}/{meth}"].append(
                            dict(old=old, new=new,
                                 revscore=1.0 - abs(new - old) / old))
                    model.cancel.rekey("app-A", version=0)

                s, c = session_split_eer(model, data, m, split, device, ds)
                if np.isfinite(c):
                    sess[f"{ds}/{meth}"].append(dict(same=s, cross=c))

    def agg(d, keys):
        out = {}
        for k, rs in d.items():
            out[k] = {kk: float(np.nanmean([r[kk] for r in rs])) for kk in keys}
            out[k]["n_seeds"] = len(rs)
        return out

    revt = agg(rev, ["old", "new", "revscore"])
    sesst = agg(sess, ["same", "cross"])
    for k, v in sesst.items():
        if np.isfinite(v["same"]) and v["same"] > 0:
            v["relative_drop_pct"] = 100.0 * (v["cross"] - v["same"]) / v["same"]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "revocability.json"), "w") as f:
        json.dump(revt, f, indent=2)
    with open(os.path.join(a.out, "cross_session.json"), "w") as f:
        json.dump(sesst, f, indent=2)

    print("=== RevScore (utility preserved after key replacement) ===")
    for k, v in sorted(revt.items()):
        print(f"  {k:28s} old={v['old']:6.2f}%  new={v['new']:6.2f}%  "
              f"RevScore={v['revscore']:.3f}  (n={v['n_seeds']})")
    print("\n=== same-session vs cross-session EER ===")
    for k, v in sorted(sesst.items()):
        print(f"  {k:28s} same={v['same']:6.2f}%  cross={v['cross']:6.2f}%  "
              f"drop={v.get('relative_drop_pct', float('nan')):6.1f}%")


if __name__ == "__main__":
    main()
