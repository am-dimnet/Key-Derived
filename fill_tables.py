"""
Measure the remaining cells of the manuscript tables.

The main grid already covers verification accuracy for every method and the
protection metrics for the protected variants.  Still missing, and computed here
so that no table cell is left unmeasured:

  * linkability (AUC and D_sys) for the unprotected CNN embedding and for the
    post-hoc CNN+BioHashing baseline;
  * the three attack settings for those two baselines and for every ablation
    variant;
  * RevScore for CNN+BioHashing;
  * same-session / cross-session EER for CNN+BioHashing and Feature+SVM.

Attack settings follow the manuscript definitions:
  template-only  the adversary holds the stored template but not the key, and
                 samples candidate templates from the empirical bit distribution
                 (for the unprotected embedding it samples from the empirical
                 per-dimension Gaussian instead);
  stolen-key     the adversary holds the key and pushes impostor ECG through the
                 genuine pipeline - a zero-effort impostor measurement;
  full-disclosure the adversary additionally holds the model and optimises a
                 representation against the stored template.
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
from metrics import (attack_success_rate, cosine_similarity, eer_from_scores,
                     hamming_similarity, linkability_auc, linkability_dsys,
                     tpr_at_far)
from models import (BioHashing, CompressiveSensing, RandomProjection,
                    derive_seed, make_transform)
from security import _binarise, embeddings, per_identity

from paths import ROOT as _ROOT


# ----------------------------------------------------------------------------

def val_threshold(scores, labels, far=0.01):
    _, thr = tpr_at_far(scores, labels, far)
    return float(thr)


def val_scores(model, data, m, device, transform=None, protected=False):
    Xe, Xq = data["beats"][m["val_enrol"]], data["beats"][m["val_query"]]
    se, sq = data["subject"][m["val_enrol"]], data["subject"][m["val_query"]]
    Ze = templates(model, Xe, device, hard=protected)
    Zq = templates(model, Xq, device, hard=protected)
    if transform is not None:
        Te, Tq = transform(Ze), transform(Zq)
        S = hamming_similarity(Tq, Te)
    elif protected:
        S = hamming_similarity(Zq.astype(np.uint8), Ze.astype(np.uint8))
    else:
        S = cosine_similarity(Zq, Ze)
    return protocols.aggregate_scores(S, se, sq)[:2]


def eer_of(model, data, m, device, transform=None, protected=False):
    Xe, Xq = data["beats"][m["enrol"]], data["beats"][m["test"]]
    se, sq = data["subject"][m["enrol"]], data["subject"][m["test"]]
    Ze = templates(model, Xe, device, hard=protected)
    Zq = templates(model, Xq, device, hard=protected)
    if transform is not None:
        S = hamming_similarity(transform(Zq), transform(Ze))
    elif protected:
        S = hamming_similarity(Zq.astype(np.uint8), Ze.astype(np.uint8))
    else:
        S = cosine_similarity(Zq, Ze)
    sc, lb, _, _ = protocols.aggregate_scores(S, se, sq)
    return eer_from_scores(sc, lb)[0]


# ----------------------------------------------------------------------------

def link_unprotected(Z, subj):
    """An unprotected embedding is identical in every application, so the mated
    cross-application score is the self-similarity."""
    ids = np.unique(subj)
    P = np.stack([Z[subj == i].mean(0) for i in ids])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ P.T
    n = len(ids)
    mated = np.diag(S)
    nonmated = S[~np.eye(n, dtype=bool)]
    return linkability_auc(mated, nonmated), linkability_dsys(mated, nonmated)


def link_biohash(Z, subj, d, L):
    Ta = BioHashing(d, L, key="app-A")(Z)
    Tb = BioHashing(d, L, key="app-B")(Z)
    Pa, ids = per_identity(Ta, subj)
    Pb, _ = per_identity(Tb, subj)
    S = hamming_similarity(Pa, Pb)
    n = len(ids)
    mated = np.diag(S)
    nonmated = S[~np.eye(n, dtype=bool)]
    return linkability_auc(mated, nonmated), linkability_dsys(mated, nonmated)


def attacks_binary(Pe, Zimp, tf, thr, model_free=True, n_attempts=200, seed=0):
    """template-only (bit sampling) and stolen-key (zero-effort impostor)."""
    rng = np.random.default_rng(seed)
    p = Pe.mean(0)
    hit = []
    for i in range(len(Pe)):
        cand = (rng.random((n_attempts, Pe.shape[1])) < p).astype(np.uint8)
        hit.append(float((hamming_similarity(cand, Pe[i:i + 1]).ravel() >= thr).any()))
    tmpl_only = 100.0 * float(np.mean(hit))
    stolen = attack_success_rate(hamming_similarity(tf(Zimp), Pe).ravel(), thr)
    return tmpl_only, stolen


def attacks_continuous(Pe, Zimp, thr, n_attempts=200, seed=0):
    """Same two settings for an unprotected embedding."""
    rng = np.random.default_rng(seed)
    mu, sd = Pe.mean(0), Pe.std(0) + 1e-8
    hit = []
    for i in range(len(Pe)):
        cand = rng.normal(mu, sd, size=(n_attempts, Pe.shape[1])).astype(np.float32)
        s = cosine_similarity(cand, Pe[i:i + 1]).ravel()
        hit.append(float((s >= thr).any()))
    tmpl_only = 100.0 * float(np.mean(hit))
    stolen = attack_success_rate(cosine_similarity(Zimp, Pe).ravel(), thr)
    return tmpl_only, stolen


def full_disclosure_binary(Pe, R, b, gamma, thr, d, steps=300, restarts=3,
                           seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Rt = torch.tensor(R, dtype=torch.float32, device=device)
    bt = torch.tensor(b, dtype=torch.float32, device=device)
    sc = float(d) ** 0.5
    succ = []
    for i in range(len(Pe)):
        target = torch.tensor(Pe[i], dtype=torch.float32, device=device) * 2 - 1
        hit = 0.0
        for r in range(restarts):
            z = torch.randn(1, d, generator=g).to(device)
            z = torch.nn.functional.normalize(z, dim=1).requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.05)
            for _ in range(steps):
                zt = torch.nn.functional.normalize(z, dim=1)
                soft = torch.tanh(gamma * (zt @ Rt.T + bt) * sc)
                loss = -(soft * target).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                hard = (soft >= 0).float().detach().cpu().numpy().astype(np.uint8)
                if float(hamming_similarity(hard, Pe[i:i + 1]).ravel()[0]) >= thr:
                    hit = 1.0
                    break
            if hit:
                break
        succ.append(hit)
    return 100.0 * float(np.mean(succ))


def full_disclosure_continuous(Pe, thr):
    """With the embedding known, the adversary simply replays it."""
    return attack_success_rate(np.diag(cosine_similarity(Pe, Pe)), thr)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", default=f"{_ROOT}/runs")
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--out", default=f"{_ROOT}/results/tables")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = defaultdict(lambda: defaultdict(list))

    for ds in a.datasets.split(","):
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        for seed in range(a.seeds):
            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)

            # ---------------- unprotected CNN embedding ----------------
            hits = sorted(glob.glob(os.path.join(a.runs, f"{ds}_cnn_L*_s{seed}",
                                                 "best.pt")))
            if hits:
                model, margs, _ = load_model(os.path.dirname(hits[0]), device)
                d = margs["d"]
                Ze = embeddings(model, data["beats"][m["enrol"]], device)
                Zq = embeddings(model, data["beats"][m["test"]], device)
                se = data["subject"][m["enrol"]]
                sq = data["subject"][m["test"]]

                sc_v, lb_v = val_scores(model, data, m, device)
                thr = val_threshold(sc_v, lb_v)
                la, dsys = link_unprotected(np.concatenate([Ze, Zq]),
                                            np.concatenate([se, sq]))
                ids = np.unique(se)
                Pe = np.stack([Ze[se == i].mean(0) for i in ids]).astype(np.float32)
                t_only, stolen = attacks_continuous(Pe, Zq, thr, seed=seed)
                fd = full_disclosure_continuous(Pe, thr)
                k = f"{ds}/cnn"
                acc[k]["link_auc"].append(la); acc[k]["link_dsys"].append(dsys)
                acc[k]["tmpl_only"].append(t_only); acc[k]["stolen"].append(stolen)
                acc[k]["full_disc"].append(fd)

                # ---------------- post-hoc protected baselines ----------------
                # BioHashing, random projection and compressive sensing all wrap
                # the same frozen encoder, so they are measured in one loop and
                # differ only in the transform object.
                L = 256
                for name, Cls in (("cnn_biohash", BioHashing),
                                  ("cnn_randproj", RandomProjection),
                                  ("cnn_compsens", CompressiveSensing)):
                    tf = Cls(d, L, key="app-A")
                    kb = f"{ds}/{name}"

                    sc_b, lb_b = val_scores(model, data, m, device, transform=tf)
                    thr_b = val_threshold(sc_b, lb_b)

                    Ta = tf(np.concatenate([Ze, Zq]))
                    Tb = Cls(d, L, key="app-B")(np.concatenate([Ze, Zq]))
                    sall = np.concatenate([se, sq])
                    Pa, ids_b = per_identity(Ta, sall)
                    Pb, _ = per_identity(Tb, sall)
                    S = hamming_similarity(Pa, Pb)
                    n_b = len(ids_b)
                    mated, nonmated = np.diag(S), S[~np.eye(n_b, dtype=bool)]
                    acc[kb]["link_auc"].append(linkability_auc(mated, nonmated))
                    acc[kb]["link_dsys"].append(linkability_dsys(mated, nonmated))

                    Peb, _ = per_identity(tf(Ze), se)
                    t_b, s_b = attacks_binary(Peb, Zq, tf, thr_b, seed=seed)
                    acc[kb]["tmpl_only"].append(t_b)
                    acc[kb]["stolen"].append(s_b)

                    # full disclosure: optimise against the known linear map
                    Wt = (tf.W.T if hasattr(tf, "W") else tf.Phi.T)
                    acc[kb]["full_disc"].append(
                        full_disclosure_binary(Peb, Wt, np.zeros(L, np.float32),
                                               1.0, thr_b, d, steps=a.steps,
                                               seed=seed, device=device))

                    # RevScore: re-key and re-measure EER
                    e_old = eer_of(model, data, m, device, transform=tf)
                    e_new = eer_of(model, data, m, device,
                                   transform=Cls(d, L, key="app-A", version=1))
                    if np.isfinite(e_old) and e_old > 0:
                        acc[kb]["rev_old"].append(e_old)
                        acc[kb]["rev_new"].append(e_new)
                        acc[kb]["revscore"].append(1 - abs(e_new - e_old) / e_old)

            # ---------------- ablation variants: attacks ----------------
            for var in ("proposed", "no_metric", "no_temp", "fixed_proj"):
                hv = sorted(glob.glob(os.path.join(
                    a.runs, f"{ds}_{var}_L*_s{seed}", "best.pt")))
                if not hv:
                    continue
                model, margs, _ = load_model(os.path.dirname(hv[0]), device)
                d, L, gamma = margs["d"], margs["L"], margs["gamma"]
                sc_v, lb_v = val_scores(model, data, m, device, protected=True)
                thr = val_threshold(sc_v, lb_v)
                Ze = embeddings(model, data["beats"][m["enrol"]], device)
                Zq = embeddings(model, data["beats"][m["test"]], device)
                se = data["subject"][m["enrol"]]
                R, b = make_transform(d, L, derive_seed("app-A", version=0))
                Pe, _ = per_identity(_binarise(Ze, R, b, gamma), se)
                tfun = lambda Z: _binarise(Z, R, b, gamma)
                t_o, st = attacks_binary(Pe, Zq, tfun, thr, seed=seed)
                fd = full_disclosure_binary(Pe, R, b, gamma, thr, d,
                                            steps=a.steps, seed=seed, device=device)
                kk = f"{ds}/{var}"
                acc[kk]["tmpl_only"].append(t_o)
                acc[kk]["stolen"].append(st)
                acc[kk]["full_disc"].append(fd)

    out = {}
    for k, v in acc.items():
        out[k] = {kk: float(np.nanmean(vv)) for kk, vv in v.items()}
        out[k]["n_seeds"] = max(len(vv) for vv in v.values())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "baseline_security.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'key':26s}{'linkAUC':>9s}{'Dsys':>8s}{'tmplOnly':>10s}"
          f"{'stolen':>9s}{'fullDisc':>10s}{'RevScore':>10s}")
    for k in sorted(out):
        v = out[k]
        print(f"{k:26s}{v.get('link_auc', float('nan')):9.3f}"
              f"{v.get('link_dsys', float('nan')):8.3f}"
              f"{v.get('tmpl_only', float('nan')):10.2f}"
              f"{v.get('stolen', float('nan')):9.2f}"
              f"{v.get('full_disc', float('nan')):10.2f}"
              f"{v.get('revscore', float('nan')):10.3f}")


if __name__ == "__main__":
    main()
