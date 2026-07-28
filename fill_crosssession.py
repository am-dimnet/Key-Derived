"""Same-session / cross-session EER for the two rows the main sweep missed:
Feature+SVM and the post-hoc CNN+BioHashing baseline, on Heartprint."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import protocols
from evaluate import feature_svm_scores, load_model, templates
from metrics import eer_from_scores, hamming_similarity
from models import BioHashing, CompressiveSensing, RandomProjection

from paths import ROOT as _ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
DS = "heartprint"
data = protocols.load(f"{_ROOT}/data/proc/{DS}.npz")
acc = defaultdict(lambda: defaultdict(list))


def same_session_masks(data, split):
    """Split each subject's Session-1 records into two halves."""
    rec = data["record"]
    by = defaultdict(list)
    for r in split["enrol_records"]:
        s = int(data["subject"][rec == r][0])
        by[s].append(int(r))
    e, q = [], []
    for s, rs in by.items():
        rs = sorted(rs)
        if len(rs) >= 2:
            e += rs[: len(rs) // 2]
            q += rs[len(rs) // 2:]
    return np.isin(rec, e), np.isin(rec, q)


for seed in range(5):
    split = protocols.make_split(data, DS, seed=seed)
    m = protocols.masks(data, split)
    me, mq = same_session_masks(data, split)

    # ---- Feature + SVM
    for tag, (a_e, a_q) in (("cross", (m["enrol"], m["test"])),
                            ("same", (me, mq))):
        mm = dict(m)
        mm["enrol"], mm["test"] = a_e, a_q
        S = feature_svm_scores(data, mm, seed=seed)
        sc, lb, _, _ = protocols.aggregate_scores(
            S, data["subject"][a_e], data["subject"][a_q])
        acc["feature_svm"][tag].append(eer_from_scores(sc, lb)[0])

    # ---- CNN + BioHashing
    hits = sorted(glob.glob(f"{_ROOT}/runs/{DS}_cnn_L*_s{seed}/best.pt"))
    if not hits:
        continue
    model, margs, _ = load_model(os.path.dirname(hits[0]), device)
    for name, Cls in (("cnn_biohash", BioHashing),
                      ("cnn_randproj", RandomProjection),
                      ("cnn_compsens", CompressiveSensing)):
        tf = Cls(margs["d"], 256, key="app-A")
        for tag, (a_e, a_q) in (("cross", (m["enrol"], m["test"])),
                                ("same", (me, mq))):
            Ze = templates(model, data["beats"][a_e], device, hard=False)
            Zq = templates(model, data["beats"][a_q], device, hard=False)
            S = hamming_similarity(tf(Zq), tf(Ze))
            sc, lb, _, _ = protocols.aggregate_scores(
                S, data["subject"][a_e], data["subject"][a_q])
            acc[name][tag].append(eer_from_scores(sc, lb)[0])

res = {}
for k, v in acc.items():
    same = float(np.nanmean(v["same"]))
    cross = float(np.nanmean(v["cross"]))
    res[k] = dict(same=same, cross=cross,
                  relative_drop_pct=100.0 * (cross - same) / same)
with open(f"{_ROOT}/results/tables/cross_session_extra.json", "w") as f:
    json.dump(res, f, indent=2)
for k, v in res.items():
    print(f"{k:16s} same={v['same']:6.2f}%  cross={v['cross']:6.2f}%  "
          f"drop={v['relative_drop_pct']:6.1f}%")
