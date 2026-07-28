"""Same-session / cross-session EER for every method on all three datasets.

Reviewer comment R2.4 asks that ECG-ID, whose enrolment and test recordings also
come from different sessions, be reported in the same format as Heartprint.  The
previous script covered Heartprint only and hard-coded the method list.

One caveat has to be measured rather than assumed.  The same-session reference
requires two disjoint recordings per subject from the *same* session.  Heartprint
provides that (Session 1 contains several records per subject).  ECG-ID and PTB
do not: in their enrolment split each subject contributes a single record, so the
same-session reference has to be formed by splitting the beats of one record,
which is an easier problem than matching two separate recordings.  The script
records which definition was used per dataset in the output, so the manuscript
can state it instead of presenting incomparable ratios as if they were alike.

Writes results/tables/cross_session_all.json.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import protocols
from evaluate import feature_svm_scores, load_model, templates
from metrics import cosine_similarity, eer_from_scores, hamming_similarity
from models import (BioHashing, CompressiveSensing, PostHocCancelable,
                    RandomProjection, derive_seed, make_transform)
from security import _binarise, embeddings

from paths import ROOT as _ROOT

R = f"{_ROOT}"
POSTHOC = {"cnn_biohash": BioHashing, "cnn_randproj": RandomProjection,
           "cnn_compsens": CompressiveSensing,
           "cnn_posthoc_ours": PostHocCancelable}


def same_session_masks(data, split):
    """Two disjoint record sets per subject from the enrolment session.

    Returns (mask_e, mask_q, definition) where definition names which of the two
    constructions was possible on this dataset.
    """
    rec = data["record"]
    by = defaultdict(list)
    for r in split["enrol_records"]:
        s = int(data["subject"][rec == r][0])
        by[s].append(int(r))

    if sum(len(v) >= 2 for v in by.values()) >= 0.5 * len(by):
        e, q = [], []
        for s, rs in by.items():
            rs = sorted(rs)
            if len(rs) >= 2:
                e += rs[: len(rs) // 2]
                q += rs[len(rs) // 2:]
        return np.isin(rec, e), np.isin(rec, q), "record-disjoint within session"

    # single enrolment record per subject: split the beats of that record
    me = np.zeros(len(rec), bool)
    mq = np.zeros(len(rec), bool)
    for s, rs in by.items():
        idx = np.where(np.isin(rec, rs))[0]
        if len(idx) < 4:
            continue
        h = len(idx) // 2
        me[idx[:h]] = True
        mq[idx[h:]] = True
    return me, mq, "beat-disjoint within one record (optimistic)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--best", default=f"{R}/results/best_configs.json")
    ap.add_argument("--out", default=f"{R}/results/tables/cross_session_all.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = json.load(open(a.best)) if os.path.exists(a.best) else {}
    out = {}

    for ds in a.datasets.split(","):
        data = protocols.load(f"{R}/data/proc/{ds}.npz")
        Lp = int(best.get(f"{ds}/proposed", {}).get("config", {}).get("L", 256))
        acc = defaultdict(lambda: defaultdict(list))
        definition = None

        for seed in range(a.seeds):
            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)
            me, mq, definition = same_session_masks(data, split)
            pairs = (("cross", (m["enrol"], m["test"])), ("same", (me, mq)))

            # ---- Feature + SVM
            for tag, (ae, aq) in pairs:
                mm = dict(m); mm["enrol"], mm["test"] = ae, aq
                S = feature_svm_scores(data, mm, seed=seed)
                sc, lb, _, _ = protocols.aggregate_scores(
                    S, data["subject"][ae], data["subject"][aq])
                acc["feature_svm"][tag].append(eer_from_scores(sc, lb)[0])

            # ---- unprotected encoders
            for meth in ("cnn", "siamese"):
                hits = sorted(glob.glob(f"{R}/runs/{ds}_{meth}_L*_s{seed}/best.pt"))
                if not hits:
                    continue
                model, margs, _ = load_model(os.path.dirname(hits[0]), device)
                for tag, (ae, aq) in pairs:
                    Ze = templates(model, data["beats"][ae], device, hard=False)
                    Zq = templates(model, data["beats"][aq], device, hard=False)
                    S = cosine_similarity(Zq, Ze)
                    sc, lb, _, _ = protocols.aggregate_scores(
                        S, data["subject"][ae], data["subject"][aq])
                    acc[meth][tag].append(eer_from_scores(sc, lb)[0])

            # ---- post-hoc protected baselines on the frozen CNN
            hits = sorted(glob.glob(f"{R}/runs/{ds}_cnn_L*_s{seed}/best.pt"))
            if hits:
                model, margs, _ = load_model(os.path.dirname(hits[0]), device)
                for name, Cls in POSTHOC.items():
                    # our own transform is the control, so it uses the template
                    # length of the proposed system; the literature baselines
                    # keep their standard 256-bit code
                    L = Lp if name == "cnn_posthoc_ours" else 256
                    tf = Cls(margs["d"], L, key="app-A")
                    for tag, (ae, aq) in pairs:
                        Ze = templates(model, data["beats"][ae], device, hard=False)
                        Zq = templates(model, data["beats"][aq], device, hard=False)
                        S = hamming_similarity(tf(Zq), tf(Ze))
                        sc, lb, _, _ = protocols.aggregate_scores(
                            S, data["subject"][ae], data["subject"][aq])
                        acc[name][tag].append(eer_from_scores(sc, lb)[0])

            # ---- proposed
            hits = sorted(glob.glob(f"{R}/runs/{ds}_proposed_L*_s{seed}/best.pt"))
            if hits:
                model, margs, _ = load_model(os.path.dirname(hits[0]), device)
                # Use the trained cancelable layer itself, exactly as
                # evaluate.py does.  Re-deriving a transform from a seed does
                # not reproduce the projection the encoder was trained against,
                # and the cross-session column would then disagree with the
                # EER reported in the main performance table.
                for tag, (ae, aq) in pairs:
                    Te = templates(model, data["beats"][ae], device, hard=True)
                    Tq = templates(model, data["beats"][aq], device, hard=True)
                    S = hamming_similarity(Tq, Te)
                    sc, lb, _, _ = protocols.aggregate_scores(
                        S, data["subject"][ae], data["subject"][aq])
                    acc["proposed"][tag].append(eer_from_scores(sc, lb)[0])

        res = {"_same_session_definition": definition}
        for k, v in acc.items():
            if not v["same"] or not v["cross"]:
                continue
            same = float(np.nanmean(v["same"]))
            cross = float(np.nanmean(v["cross"]))
            res[k] = dict(same=same, cross=cross,
                          relative_drop_pct=100.0 * (cross - same) / max(same, 1e-9),
                          n_seeds=len(v["same"]))
        out[ds] = res
        print(f"=== {ds}  ({definition}) ===", flush=True)
        for k in sorted(res):
            if k.startswith("_"):
                continue
            r = res[k]
            print(f"  {k:20s} same={r['same']:7.2f}  cross={r['cross']:7.2f}  "
                  f"drop={r['relative_drop_pct']:8.1f}%", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
