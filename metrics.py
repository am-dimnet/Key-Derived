"""
Verification, template-protection and attack metrics.

Everything downstream consumes a *score file*: arrays of scores together with the
genuine/impostor label and the subject id of the probe.  Keeping that artefact is
what makes TPR@FAR, DET curves, subject-level bootstrap intervals and the paired
Wilcoxon test reproducible (reviewer comment R1.12).
"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------

def eer_from_scores(scores, labels):
    """Equal error rate (%) and the threshold at which it is attained."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return 100.0 * float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def auc_from_scores(scores, labels):
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, float)))


def tpr_at_far(scores, labels, far=0.01):
    """TPR at a fixed FAR operating point (%)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, far, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return 100.0 * float(tpr[idx]), float(thr[idx])


def far_frr_at(scores, labels, threshold):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    imp, gen = scores[labels == 0], scores[labels == 1]
    far = 100.0 * float((imp >= threshold).mean()) if imp.size else float("nan")
    frr = 100.0 * float((gen < threshold).mean()) if gen.size else float("nan")
    return far, frr


def det_curve(scores, labels):
    fpr, tpr, _ = roc_curve(np.asarray(labels, int), np.asarray(scores, float))
    return fpr, 1.0 - tpr


# ----------------------------------------------------------------------------
# uncertainty: subject-level bootstrap  (R1.10, R1.12)
# ----------------------------------------------------------------------------

def subject_bootstrap_ci(scores, labels, subjects, fn=eer_from_scores,
                         n_boot=1000, alpha=0.05, seed=0):
    """Resample *subjects* (not trials) and recompute the statistic."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    index = {s: np.flatnonzero(subjects == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([index[s] for s in pick])
        v = fn(scores[sel], labels[sel])
        v = v[0] if isinstance(v, tuple) else v
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_subject_metric(scores, labels, subjects, fn=eer_from_scores):
    """Statistic computed separately for every probe subject.

    These are the paired observations used by the Wilcoxon test (R1.12): one
    value per subject per system, on an identical subject set.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    subjects = np.asarray(subjects)
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        if len(np.unique(labels[m])) < 2:
            continue
        v = fn(scores[m], labels[m])
        out[s] = v[0] if isinstance(v, tuple) else v
    return out


def paired_wilcoxon(a: dict, b: dict):
    """Signed-rank test over the subjects present in both systems."""
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys], float)
    y = np.array([b[k] for k in keys], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.allclose(x, y):
        return dict(n_pairs=int(len(x)), statistic=float("nan"), p_value=float("nan"))
    st, p = wilcoxon(x, y)
    return dict(n_pairs=int(len(x)), statistic=float(st), p_value=float(p),
                median_diff=float(np.median(x - y)))


# ----------------------------------------------------------------------------
# template protection
# ----------------------------------------------------------------------------

def hamming_similarity(Tq, Te):
    """s = 1 - d_H / L for every query row against every enrolled row."""
    Tq = np.asarray(Tq, np.uint8)
    Te = np.asarray(Te, np.uint8)
    L = Tq.shape[1]
    # (a != b) count via matrix products on +-1 codes
    A = 2.0 * Tq.astype(np.float32) - 1.0
    B = 2.0 * Te.astype(np.float32) - 1.0
    agree = (A @ B.T + L) / 2.0
    return agree / L


def cosine_similarity(Zq, Ze):
    Zq = np.asarray(Zq, np.float32)
    Ze = np.asarray(Ze, np.float32)
    Zq = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
    Ze = Ze / (np.linalg.norm(Ze, axis=1, keepdims=True) + 1e-8)
    return Zq @ Ze.T


def revscore(eer_old, eer_new):
    """Utility preservation after re-issuance.  NOT a security measure (R1.7)."""
    if not np.isfinite(eer_old) or eer_old <= 0:
        return float("nan")
    return float(1.0 - abs(eer_new - eer_old) / eer_old)


def linkability_auc(mated_scores, nonmated_scores):
    """AUC of separating cross-key mated from cross-key non-mated pairs.

    0.5 means the two are indistinguishable, i.e. templates issued under
    different keys cannot be linked.
    """
    s = np.concatenate([np.asarray(mated_scores, float),
                        np.asarray(nonmated_scores, float)])
    y = np.concatenate([np.ones(len(mated_scores), int),
                        np.zeros(len(nonmated_scores), int)])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def linkability_dsys(mated, nonmated, omega=1.0, nbins=100):
    """System-level linkability D_sys of Gomez-Barrero et al. (2017).

    0 = fully unlinkable, 1 = fully linkable.  Reported alongside the AUC because
    reviewer comment R1.12 asks for the standard framework from the
    template-protection literature.
    """
    mated = np.asarray(mated, float)
    nonmated = np.asarray(nonmated, float)
    if mated.size == 0 or nonmated.size == 0:
        return float("nan")
    lo = min(mated.min(), nonmated.min())
    hi = max(mated.max(), nonmated.max())
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, nbins + 1)
    pm, _ = np.histogram(mated, bins=edges, density=False)
    pn, _ = np.histogram(nonmated, bins=edges, density=False)
    pm = pm / max(pm.sum(), 1)
    pn = pn / max(pn.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(pn > 0, pm / pn, np.inf)
    d = np.zeros_like(pm, dtype=float)
    m = omega * lr
    good = np.isfinite(m) & (m > 1)
    d[good] = 2.0 * m[good] / (1.0 + m[good]) - 1.0
    d[np.isinf(lr) & (pm > 0)] = 1.0
    return float(np.sum(pm * d))


def attack_success_rate(attack_scores, threshold):
    """ASR (%) at a threshold fixed on the clean validation set."""
    a = np.asarray(attack_scores, float)
    if a.size == 0:
        return float("nan")
    return 100.0 * float((a >= threshold).mean())


# ----------------------------------------------------------------------------
# convenience
# ----------------------------------------------------------------------------

def summarise(scores, labels, subjects=None, far=0.01, n_boot=500, seed=0):
    eer, thr = eer_from_scores(scores, labels)
    auc = auc_from_scores(scores, labels)
    tpr, _ = tpr_at_far(scores, labels, far)
    out = dict(eer=eer, auc=auc, tpr_at_far1=tpr, eer_threshold=thr,
               n_genuine=int(np.sum(np.asarray(labels) == 1)),
               n_impostor=int(np.sum(np.asarray(labels) == 0)))
    if subjects is not None:
        lo, hi = subject_bootstrap_ci(scores, labels, subjects,
                                      eer_from_scores, n_boot=n_boot, seed=seed)
        out["eer_ci95"] = [lo, hi]
    return out
