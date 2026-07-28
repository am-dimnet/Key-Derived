"""
Template-protection and attack experiments.

Threat model (reviewer comment R1.4).  The secret is a 256-bit application key
that seeds the projection and bias; templates are additionally indexed by a
version number.  Under an application-level key every enrolled user in one
deployment shares the key, so the disclosure settings separate as follows:

  A. template-only     adversary holds the stored protected template, not the key
                       and not the model.  It cannot transform its own ECG, so it
                       can only (a) replay the stolen template verbatim, or
                       (b) sample candidate templates from the empirical bit
                       distribution of leaked templates.
  B. stolen-key        adversary additionally holds the application key.  It
                       transforms its own (impostor) ECG through the genuine
                       pipeline.  This measurement is a zero-effort impostor
                       false-acceptance rate and is reported as such (R1.6).
  C. full-disclosure   adversary additionally holds the trained model weights and
                       optimises a soft template by gradient descent against the
                       stolen template.  Budget and stopping rule are reported.

The threshold is fixed on the validation subjects and never adjusted for attacks.
"""
import argparse
import json
import os

import numpy as np
import torch

import protocols
from evaluate import load_model, templates
from metrics import (attack_success_rate, eer_from_scores, hamming_similarity,
                     linkability_auc, linkability_dsys, revscore, tpr_at_far)
from models import make_transform, derive_seed

from paths import ROOT as _ROOT


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _binarise(Z, R, b, gamma):
    # same sqrt(d) rescaling as CancelableLayer.forward
    U = (Z @ R.T + b) * (R.shape[1] ** 0.5)
    return (np.tanh(gamma * U) >= 0).astype(np.uint8)


def embeddings(model, X, device, bs=1024):
    """Unprotected embeddings z (before the cancelable layer)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(device)
            out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out)


def per_identity(T, subj):
    """One mean template per identity, re-binarised (the enrolled record)."""
    ids = np.unique(subj)
    out = np.stack([(T[subj == i].mean(0) >= 0.5).astype(np.uint8) for i in ids])
    return out, ids


def validation_threshold(model, data, m, device, protected, far=0.01):
    """tau fixed on the validation subjects, at both EER and FAR=1%."""
    from metrics import cosine_similarity
    Xe, Xq = data["beats"][m["val_enrol"]], data["beats"][m["val_query"]]
    se, sq = data["subject"][m["val_enrol"]], data["subject"][m["val_query"]]
    Te = templates(model, Xe, device, hard=protected)
    Tq = templates(model, Xq, device, hard=protected)
    S = (hamming_similarity(Tq.astype(np.uint8), Te.astype(np.uint8))
         if protected else cosine_similarity(Tq, Te))
    sc, lb, _, _ = protocols.aggregate_scores(S, se, sq)
    _, thr_eer = eer_from_scores(sc, lb)
    _, thr_far = tpr_at_far(sc, lb, far)
    return float(thr_eer), float(thr_far)


# ----------------------------------------------------------------------------
# unlinkability  (R1.12 / C3)
# ----------------------------------------------------------------------------

def unlinkability(Z, subj, d, L, gamma, keys=("app-A", "app-B"), n_pairs=20000,
                  seed=0):
    """Cross-key mated vs non-mated similarity -> AUC and D_sys."""
    rng = np.random.default_rng(seed)
    Ra, ba = make_transform(d, L, derive_seed(keys[0], version=0, bind_user=False))
    Rb, bb = make_transform(d, L, derive_seed(keys[1], version=0, bind_user=False))
    Ta = _binarise(Z, Ra, ba, gamma)
    Tb = _binarise(Z, Rb, bb, gamma)
    Pa, ids = per_identity(Ta, subj)
    Pb, _ = per_identity(Tb, subj)
    S = hamming_similarity(Pa, Pb)
    n = len(ids)
    mated = np.diag(S)
    off = ~np.eye(n, dtype=bool)
    nonmated = S[off]
    if len(nonmated) > n_pairs:
        nonmated = rng.choice(nonmated, n_pairs, replace=False)
    return dict(link_auc=linkability_auc(mated, nonmated),
                link_dsys=linkability_dsys(mated, nonmated),
                n_identities=int(n),
                mated_mean=float(mated.mean()), nonmated_mean=float(nonmated.mean()))


# ----------------------------------------------------------------------------
# revocation security  (R1.7)
# ----------------------------------------------------------------------------

def revocation(Ze, Zq, subj_e, subj_q, d, L, gamma, thr, app_key="app-A",
               n_versions=6, seed=0):
    """(a) do superseded templates still authenticate,
       (b) does observing k revoked templates raise linkability,
       (c) does compromising version v help against version v+1."""
    rng = np.random.default_rng(seed)
    ver = {}
    for v in range(n_versions):
        R, b = make_transform(d, L, derive_seed(app_key, version=v, bind_user=False))
        Pe, ids = per_identity(_binarise(Ze, R, b, gamma), subj_e)
        Tq = _binarise(Zq, R, b, gamma)
        ver[v] = dict(R=R, b=b, Pe=Pe, ids=ids, Tq=Tq)

    # ---- (a) old-version query against re-issued enrolment
    S_old_new = hamming_similarity(ver[0]["Tq"], ver[1]["Pe"])
    sc, lb, _, _ = protocols.aggregate_scores(S_old_new, ver[1]["ids"], subj_q)
    gen_old = sc[lb == 1]
    acc_stale = attack_success_rate(gen_old, thr)
    # matched-version reference, i.e. what a legitimate user achieves
    S_new_new = hamming_similarity(ver[1]["Tq"], ver[1]["Pe"])
    sc2, lb2, _, _ = protocols.aggregate_scores(S_new_new, ver[1]["ids"], subj_q)
    acc_fresh = attack_success_rate(sc2[lb2 == 1], thr)
    # chance level: impostor trials under the matched version
    acc_chance = attack_success_rate(sc2[lb2 == 0], thr)

    # ---- (b) linkability given k revoked templates of the same subject
    curve = []
    target = ver[n_versions - 1]["Pe"]
    for k in range(1, n_versions):
        best_m, best_n = [], []
        for i in range(len(target)):
            revoked = np.stack([ver[v]["Pe"][i] for v in range(k)])
            s_self = hamming_similarity(revoked, target[i:i + 1]).max()
            others = np.setdiff1d(np.arange(len(target)), [i])
            s_other = hamming_similarity(revoked, target[others]).max(0)
            best_m.append(float(s_self))
            best_n.extend(s_other.tolist())
        curve.append(dict(k=k, link_auc=linkability_auc(best_m, best_n),
                          link_dsys=linkability_dsys(np.array(best_m),
                                                     np.array(best_n))))

    # ---- (c) transfer of a compromised version to the renewed one
    #      the adversary knows version 0 templates and tries them on version 1
    S_transfer = hamming_similarity(ver[0]["Pe"], ver[1]["Pe"])
    transfer = attack_success_rate(np.diag(S_transfer), thr)

    return dict(stale_template_acceptance_pct=acc_stale,
                fresh_template_acceptance_pct=acc_fresh,
                impostor_acceptance_pct=acc_chance,
                linkability_vs_k=curve,
                cross_version_transfer_pct=transfer)


# ----------------------------------------------------------------------------
# attacks  (R1.6)
# ----------------------------------------------------------------------------

def attack_template_only(Pe, thr, n_attempts=200, seed=0):
    """No key, no model.  (a) replay the stolen template, (b) sample bits."""
    rng = np.random.default_rng(seed)
    replay = attack_success_rate(np.diag(hamming_similarity(Pe, Pe)), thr)
    p = Pe.mean(0)                                   # empirical per-bit rate
    succ = []
    for i in range(len(Pe)):
        cand = (rng.random((n_attempts, Pe.shape[1])) < p).astype(np.uint8)
        s = hamming_similarity(cand, Pe[i:i + 1]).ravel()
        succ.append(float((s >= thr).any()))
    return dict(replay_stolen_template_pct=replay,
                bit_sampling_pct=100.0 * float(np.mean(succ)),
                attempts_per_identity=int(n_attempts))


def attack_stolen_key(Zimp, Pe, R, b, gamma, thr):
    """Key known: push impostor ECG through the genuine pipeline.

    This is a zero-effort impostor false-acceptance measurement (R1.6) and is
    labelled as such wherever it is reported.
    """
    Timp = _binarise(Zimp, R, b, gamma)
    S = hamming_similarity(Timp, Pe)
    return dict(zero_effort_impostor_far_pct=attack_success_rate(S.ravel(), thr),
                n_trials=int(S.size))


def attack_full_disclosure(Pe, R, b, gamma, thr, d, steps=300, lr=0.05,
                           restarts=3, seed=0, device="cpu"):
    """Model and key known: optimise an embedding so its template matches.

    Budget: `restarts` random restarts x `steps` Adam iterations, stopping early
    once the surrogate similarity exceeds the threshold.  Both are reported so the
    attack strength is quantifiable.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Rt = torch.tensor(R, dtype=torch.float32, device=device)
    bt = torch.tensor(b, dtype=torch.float32, device=device)
    succ, iters = [], []
    for i in range(len(Pe)):
        target = torch.tensor(Pe[i], dtype=torch.float32, device=device) * 2 - 1
        hit, used = 0.0, steps * restarts
        for r in range(restarts):
            z = torch.randn(1, d, generator=g).to(device)
            z = torch.nn.functional.normalize(z, dim=1).requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            for t in range(steps):
                zt = torch.nn.functional.normalize(z, dim=1)
                soft = torch.tanh(gamma * (zt @ Rt.T + bt))
                loss = -(soft * target).mean()          # surrogate distance
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    hard = (soft >= 0).float().cpu().numpy().astype(np.uint8)
                    s = float(hamming_similarity(hard, Pe[i:i + 1]).ravel()[0])
                if s >= thr:
                    hit, used = 1.0, r * steps + t + 1
                    break
            if hit:
                break
        succ.append(hit); iters.append(used)
    return dict(full_disclosure_asr_pct=100.0 * float(np.mean(succ)),
                mean_iterations=float(np.mean(iters)),
                budget=dict(restarts=restarts, steps=steps, lr=lr,
                            stopping_rule="stop when hard-template similarity >= tau"))


# ----------------------------------------------------------------------------
# attribute leakage  (R2.1)
# ----------------------------------------------------------------------------

def attribute_leakage(Z, T, subj, records, meta_path, dataset, seed=0):
    """Can age / sex / diagnosis be inferred from the embedding vs the template?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    with open(meta_path) as f:
        meta = json.load(f)
    rec_meta = meta["record_meta"]

    targets = {}
    if dataset == "ecgid":
        sex = np.array([str(rec_meta[r].get("sex") or "") for r in records])
        age = np.array([rec_meta[r].get("age") or -1 for r in records], float)
        if (sex != "").sum() > 50:
            targets["sex"] = (sex == "male").astype(int), sex != ""
        if (age > 0).sum() > 50:
            med = np.median(age[age > 0])
            targets["age_above_median"] = (age > med).astype(int), age > 0
    elif dataset == "ptbdb":
        dx = np.array([str(rec_meta[r].get("diagnosis") or "") for r in records])
        ok = dx != ""
        if ok.sum() > 50:
            healthy = np.char.find(dx.astype(str), "healthy") >= 0
            targets["healthy_vs_disease"] = healthy.astype(int), ok

    out = {}
    for name, (y, ok) in targets.items():
        row = {}
        for space, X in (("unprotected_embedding", Z), ("protected_template", T)):
            Xa, ya, ga = X[ok], y[ok], subj[ok]
            if len(np.unique(ya)) < 2 or len(np.unique(ga)) < 4:
                continue
            n_splits = min(4, len(np.unique(ga)))
            gkf = GroupKFold(n_splits=n_splits)
            preds, trues = [], []
            for tr, te in gkf.split(Xa, ya, ga):
                if len(np.unique(ya[tr])) < 2:
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(Xa[tr], ya[tr])
                preds.append(clf.predict(Xa[te])); trues.append(ya[te])
            if preds:
                row[space] = 100.0 * float(balanced_accuracy_score(
                    np.concatenate(trues), np.concatenate(preds)))
        row["chance_pct"] = 50.0
        row["n_samples"] = int(ok.sum())
        out[name] = row
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", default="proposed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--runs", default=f"{_ROOT}/runs")
    ap.add_argument("--out", default=f"{_ROOT}/results")
    ap.add_argument("--attack_steps", type=int, default=300)
    ap.add_argument("--skip_attacks", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = protocols.load(os.path.join(a.proc, f"{a.dataset}.npz"))
    split = protocols.make_split(data, a.dataset, seed=a.seed)
    m = protocols.masks(data, split)

    run_dir = os.path.join(a.runs, f"{a.dataset}_{a.variant}_L{a.L}_s{a.seed}")
    model, margs, _ = load_model(run_dir, device)
    d, L, gamma = margs["d"], margs["L"], margs["gamma"]
    protected = model.cancel is not None

    thr_eer, thr_far = validation_threshold(model, data, m, device, protected)
    print(f"validation thresholds: EER-op={thr_eer:.4f}  FAR1%-op={thr_far:.4f}",
          flush=True)

    Ze = embeddings(model, data["beats"][m["enrol"]], device)
    Zq = embeddings(model, data["beats"][m["test"]], device)
    se = data["subject"][m["enrol"]]
    sq = data["subject"][m["test"]]

    res = dict(dataset=a.dataset, variant=a.variant, seed=a.seed, L=L,
               threshold_eer_op=thr_eer, threshold_far1_op=thr_far,
               threshold_source="validation subjects, never adjusted for attacks")

    # ---- unlinkability
    res["unlinkability"] = unlinkability(np.concatenate([Ze, Zq]),
                                         np.concatenate([se, sq]),
                                         d, L, gamma, seed=a.seed)
    print("unlinkability:", json.dumps(res["unlinkability"]), flush=True)

    # ---- revocation security
    res["revocation"] = revocation(Ze, Zq, se, sq, d, L, gamma, thr_far,
                                   seed=a.seed)
    print("revocation:", json.dumps({k: v for k, v in res["revocation"].items()
                                     if k != "linkability_vs_k"}), flush=True)

    if not a.skip_attacks:
        R, b = make_transform(d, L, derive_seed("app-A", version=0, bind_user=False))
        Pe, ids = per_identity(_binarise(Ze, R, b, gamma), se)
        # impostor pool: query beats of every other identity
        res["attack_template_only"] = attack_template_only(Pe, thr_far, seed=a.seed)
        res["attack_stolen_key"] = attack_stolen_key(Zq, Pe, R, b, gamma, thr_far)
        res["attack_full_disclosure"] = attack_full_disclosure(
            Pe, R, b, gamma, thr_far, d, steps=a.attack_steps,
            seed=a.seed, device=device)
        for k in ("attack_template_only", "attack_stolen_key",
                  "attack_full_disclosure"):
            print(f"{k}: {json.dumps(res[k])}", flush=True)

    # ---- attribute leakage
    try:
        Tall = np.concatenate([
            templates(model, data["beats"][m["enrol"]], device, hard=True),
            templates(model, data["beats"][m["test"]], device, hard=True)])
        recs = np.concatenate([data["record"][m["enrol"]], data["record"][m["test"]]])
        res["attribute_leakage"] = attribute_leakage(
            np.concatenate([Ze, Zq]), Tall, np.concatenate([se, sq]), recs,
            os.path.join(a.proc, f"{a.dataset}_meta.json"), a.dataset, seed=a.seed)
        print("attribute_leakage:", json.dumps(res["attribute_leakage"]), flush=True)
    except Exception as e:
        res["attribute_leakage_error"] = str(e)
        print("attribute leakage failed:", e, flush=True)

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.variant}_L{L}_s{a.seed}"
    with open(os.path.join(a.out, f"security_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] security results written", flush=True)


if __name__ == "__main__":
    main()
