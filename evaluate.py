"""
Protocol evaluation.

Every method produces the same artefact: a score file holding one row per
(query beat, enrolled identity) trial with its score, genuine/impostor label and
probe subject.  Tables, DET curves, bootstrap intervals and the paired Wilcoxon
test are all computed from that file, so the reported numbers can be recomputed
without retraining (reviewer comment R1.12).

Methods
  proposed            jointly trained encoder + key-derived cancelable layer
  cnn                 unprotected deep embedding, cosine matching
  siamese             unprotected embedding trained with the metric loss only
  cnn_biohash         frozen CNN embedding + BioHashing        (post-hoc baseline)
  cnn_posthoc_ours    frozen CNN embedding + our transform     (control for R1.1)
  feature_svm         handcrafted features + SVM on pair differences
  no_metric / no_temp / fixed_proj      ablation variants
"""
import argparse
import json
import os

import numpy as np
import torch

import protocols
from metrics import (auc_from_scores, eer_from_scores, cosine_similarity,
                     hamming_similarity, per_subject_metric, subject_bootstrap_ci,
                     tpr_at_far)
from models import (BioHashing, CompressiveSensing, PostHocCancelable,
                    ProtectedNet, RandomProjection)

from paths import ROOT as _ROOT

PROTECTED = {"proposed", "no_metric", "no_temp", "fixed_proj"}
POSTHOC = {"cnn_biohash": BioHashing,
           "cnn_randproj": RandomProjection,
           "cnn_compsens": CompressiveSensing,
           "cnn_posthoc_ours": PostHocCancelable}


# ----------------------------------------------------------------------------
# handcrafted-feature baseline
# ----------------------------------------------------------------------------

def handcrafted(B):
    """Fiducial-ish + statistical + spectral descriptors of a 512-sample beat."""
    from scipy.stats import kurtosis, skew
    from scipy.fft import dct
    B = np.asarray(B, np.float32)
    R = 200                                    # R peak sits at PRE by construction
    feats = [
        B.mean(1), B.std(1), skew(B, axis=1), kurtosis(B, axis=1),
        np.sqrt((B ** 2).mean(1)),
        B.max(1), B.min(1), B.argmax(1).astype(np.float32), B.argmin(1).astype(np.float32),
        B[:, R],                                                   # R amplitude
        B[:, R - 40:R].min(1), B[:, R:R + 40].min(1),              # Q and S troughs
        B[:, R - 120:R - 40].max(1),                               # P wave
        B[:, R + 60:R + 260].max(1),                               # T wave
        (B[:, R - 40:R].argmin(1) - 40).astype(np.float32),        # QR interval
        B[:, R:R + 40].argmin(1).astype(np.float32),               # RS interval
        (B[:, R + 60:R + 260].argmax(1) + 60).astype(np.float32),  # RT interval
    ]
    F = np.stack(feats, axis=1)
    D = dct(B, axis=1, norm="ortho")[:, :32]   # non-fiducial DCT descriptor
    F = np.concatenate([F, D], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def feature_svm_scores(data, m, seed=0):
    """Linear SVM over |f_query - f_enrol| pair differences."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    Ftr = handcrafted(data["beats"][m["train"]])
    ytr = data["subject"][m["train"]]
    sc = StandardScaler().fit(Ftr)
    Ftr = sc.transform(Ftr)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ftr))[:6000]
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    X = np.abs(Ftr[a] - Ftr[b])
    y = (ytr[a] == ytr[b]).astype(int)
    if y.sum() < 20:                            # ensure some genuine pairs exist
        pos = []
        for s in np.unique(ytr):
            w = np.flatnonzero(ytr == s)
            if len(w) >= 2:
                p = rng.choice(w, size=(min(60, len(w) // 2), 2), replace=True)
                pos.append(np.abs(Ftr[p[:, 0]] - Ftr[p[:, 1]]))
        if pos:
            P = np.concatenate(pos)
            X = np.concatenate([X, P])
            y = np.concatenate([y, np.ones(len(P), int)])
    clf = LinearSVC(C=0.1, max_iter=5000).fit(X, y)

    Fe = sc.transform(handcrafted(data["beats"][m["enrol"]]))
    Fq = sc.transform(handcrafted(data["beats"][m["test"]]))
    # score = -distance in the SVM's learned space, computed pairwise
    S = np.empty((len(Fq), len(Fe)), np.float32)
    for i in range(0, len(Fq), 64):
        blk = Fq[i:i + 64]
        d = np.abs(blk[:, None, :] - Fe[None, :, :]).reshape(-1, Fe.shape[1])
        S[i:i + 64] = clf.decision_function(d).reshape(len(blk), len(Fe))
    return S


# ----------------------------------------------------------------------------
# deep methods
# ----------------------------------------------------------------------------

def load_model(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    a = ck["args"]
    from train import VARIANTS
    cfg = VARIANTS[a["variant"]]
    model = ProtectedNet(n_classes=len(ck["classes"]), d=a["d"], L=a["L"],
                         gamma=a["gamma"],
                         app_key=(a["app_key"] if cfg["keyed"] else "FIXED-PROJECTION"),
                         use_cancelable=cfg["cancel"],
                         arch=a.get("arch", "plain"), head=a.get("head", "linear"),
                         arc_s=a.get("arc_s", 30.0),
                         arc_m=a.get("arc_m", 0.30),
                         drop=a.get("drop", 0.1)).to(device)
    model.load_state_dict(ck["model"])
    if model.cancel is not None and ck.get("gamma_at_best"):
        model.cancel.gamma = float(ck["gamma_at_best"])   # match training-time steepness
    model.eval()
    return model, a, ck


@torch.no_grad()
def templates(model, X, device, hard, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(device)
        out.append(model.template(xb, hard=hard).cpu().numpy())
    return np.concatenate(out)


def score_matrix(method, data, m, run_dir, device, L=256, app_key="app-A",
                 seed=0, user_key=""):
    if method == "feature_svm":
        return feature_svm_scores(data, m, seed=seed), None

    model, a, _ = load_model(run_dir, device)
    Xe, Xq = data["beats"][m["enrol"]], data["beats"][m["test"]]

    if method in PROTECTED:
        Te = templates(model, Xe, device, hard=True).astype(np.uint8)
        Tq = templates(model, Xq, device, hard=True).astype(np.uint8)
        return hamming_similarity(Tq, Te), (Tq, Te)

    Ze = templates(model, Xe, device, hard=False)
    Zq = templates(model, Xq, device, hard=False)

    if method in POSTHOC:
        tf = POSTHOC[method](d=Ze.shape[1], L=L, key=app_key, user=user_key)
        Te, Tq = tf(Ze), tf(Zq)
        return hamming_similarity(Tq, Te), (Tq, Te)

    return cosine_similarity(Zq, Ze), (Zq, Ze)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def evaluate(dataset, method, seed, proc, runs, out, L=256, app_key="app-A",
             n_boot=500, tag_override="", suffix=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = protocols.load(os.path.join(proc, f"{dataset}.npz"))
    split = protocols.make_split(data, dataset, seed=seed)
    m = protocols.masks(data, split)

    # which trained run supplies the encoder.  The tuned configuration decides
    # the template length, so the directory is located by globbing rather than
    # assuming the default L.
    backbone = {"cnn_biohash": "cnn", "cnn_randproj": "cnn",
                "cnn_compsens": "cnn", "cnn_posthoc_ours": "cnn"}.get(method, method)
    run_dir = (os.path.join(runs, tag_override) if tag_override
               else os.path.join(runs, f"{dataset}_{backbone}_L{L}_s{seed}"))
    if method != "feature_svm" and not os.path.exists(os.path.join(run_dir, "best.pt")):
        import glob as _glob
        hits = sorted(_glob.glob(os.path.join(
            runs, f"{dataset}_{backbone}_L*_s{seed}", "best.pt")))
        if not hits:
            raise FileNotFoundError(f"missing checkpoint for {run_dir}")
        run_dir = os.path.dirname(hits[0])
        # For post-hoc methods L is a free parameter of the transform, not a
        # property of the backbone checkpoint, so the requested L must survive.
        # Overwriting it here silently measured the ECG-ID control at L=256
        # while the proposed system ran at L=512, breaking the matched
        # comparison the control exists to provide (R1.1).
        if method not in POSTHOC:
            L = int(os.path.basename(run_dir).split("_L")[1].split("_")[0])

    S, _ = score_matrix(method, data, m, run_dir, device, L=L,
                        app_key=app_key, seed=seed)
    sc, lb, probe, ids = protocols.aggregate_scores(
        S, data["subject"][m["enrol"]], data["subject"][m["test"]])

    eer, thr = eer_from_scores(sc, lb)
    auc = auc_from_scores(sc, lb)
    tpr, thr_far = tpr_at_far(sc, lb, 0.01)
    lo, hi = subject_bootstrap_ci(sc, lb, probe, eer_from_scores,
                                  n_boot=n_boot, seed=seed)
    per_subj = per_subject_metric(sc, lb, probe, eer_from_scores)

    os.makedirs(out, exist_ok=True)
    tag = f"{dataset}_{method}_L{L}_s{seed}{suffix}"
    np.savez_compressed(os.path.join(out, f"scores_{tag}.npz"),
                        score=sc.astype(np.float32), label=lb.astype(np.int8),
                        probe_subject=probe.astype(np.int32),
                        enrolled_ids=ids.astype(np.int32))
    res = dict(dataset=dataset, method=method, seed=seed, L=L,
               eer=eer, auc=auc, tpr_at_far1=tpr,
               eer_ci95=[lo, hi], eer_threshold=thr, far1_threshold=thr_far,
               n_trials=int(len(sc)), n_genuine=int(lb.sum()),
               n_impostor=int(len(lb) - lb.sum()),
               n_enrolled_identities=int(len(ids)),
               per_subject_eer={int(k): float(v) for k, v in per_subj.items()})
    with open(os.path.join(out, f"result_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] EER={eer:.3f}% (95% CI {lo:.2f}-{hi:.2f})  AUC={auc:.4f}  "
          f"TPR@FAR1%={tpr:.2f}%  trials={len(sc)}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--tag", default="",
                    help="explicit run directory name under --runs")
    ap.add_argument("--suffix", default="",
                    help="appended to the result/scores filename stem")
    ap.add_argument("--runs", default=f"{_ROOT}/runs")
    ap.add_argument("--out", default=f"{_ROOT}/results")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--app_key", default="app-A")
    a = ap.parse_args()
    evaluate(a.dataset, a.method, a.seed, a.proc, a.runs, a.out, L=a.L,
             app_key=a.app_key, tag_override=a.tag, suffix=a.suffix)
