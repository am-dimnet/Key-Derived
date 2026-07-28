"""
Training driver.

Objective (manuscript Eq. 12):   L = L_cls + l1 * L_metric + l2 * L_temp

  L_cls     cross entropy on the soft protected template
  L_metric  contrastive loss on normalised soft-template distances
  L_temp    bit-balance term + bit-decorrelation term

Note on scaling: the manuscript writes the decorrelation term as a Frobenius norm
(a sum over L^2 entries).  We use the mean over those entries instead, which only
rescales lambda_3 and keeps the term numerically comparable to the others.

Variants (`--variant`) cover the baselines and the ablation grid in one place:

  proposed        cancelable layer + all three losses
  cnn             no cancelable layer, classification only (unprotected embedding)
  siamese         no cancelable layer, metric loss only  (unprotected embedding)
  no_metric       cancelable layer, no metric loss
  no_temp         cancelable layer, no template regularisation
  fixed_proj      cancelable layer whose projection is NOT key-derived
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# Several trainings share one GPU during the hyper-parameter search; without a
# thread cap each process grabs all 22 cores and they thrash each other.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

import protocols
from models import ProtectedNet

from paths import ROOT as _ROOT

VARIANTS = {
    #                 cancelable  cls  metric  temp   key-derived
    "proposed":  dict(cancel=True,  cls=True,  metric=True,  temp=True,  keyed=True),
    "cnn":       dict(cancel=False, cls=True,  metric=False, temp=False, keyed=False),
    "siamese":   dict(cancel=False, cls=False, metric=True,  temp=False, keyed=False),
    "no_metric": dict(cancel=True,  cls=True,  metric=False, temp=True,  keyed=True),
    "no_temp":   dict(cancel=True,  cls=True,  metric=True,  temp=False, keyed=True),
    "fixed_proj": dict(cancel=True, cls=True,  metric=True,  temp=True,  keyed=False),
}


# ----------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------

def metric_loss(T, y, margin=1.0):
    """Contrastive loss on d_ij = ||T_i - T_j||_2 / sqrt(L)."""
    L = T.shape[1]
    d = torch.cdist(T, T, p=2) / (L ** 0.5)
    same = (y[:, None] == y[None, :])
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = same & ~eye
    neg = ~same
    lp = (d[pos] ** 2).mean() if pos.any() else T.sum() * 0.0
    ln = (F.relu(margin - d[neg]) ** 2).mean() if neg.any() else T.sum() * 0.0
    return lp + ln


def triplet_loss(T, y, margin=0.5):
    """Batch-hard triplet on normalised soft-template distances.

    For every anchor the hardest positive and hardest negative inside the batch
    are used, which gives a much stronger training signal than averaging over all
    pairs once the easy pairs are already separated.
    """
    L = T.shape[1]
    d = torch.cdist(T, T, p=2) / (L ** 0.5)
    same = (y[:, None] == y[None, :])
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = same & ~eye
    neg = ~same
    big = torch.finfo(d.dtype).max
    hardest_pos = torch.where(pos, d, torch.full_like(d, -1.0)).max(1).values
    hardest_neg = torch.where(neg, d, torch.full_like(d, big)).min(1).values
    ok = pos.any(1) & neg.any(1)
    if not ok.any():
        return T.sum() * 0.0
    return F.relu(hardest_pos[ok] - hardest_neg[ok] + margin).mean()


def hamming_metric_loss(T, y, margin=0.3):
    """Matching-aware loss on the soft template.

    Authentication decides with Hamming similarity on the binarised template, so
    the training signal should shape exactly that quantity.  For soft codes in
    [-1,1] the normalised inner product s_ij = <T_i, T_j> / L is the differentiable
    counterpart of 1 - 2*d_H/L: it is +1 when every bit agrees and -1 when every
    bit differs.  Genuine pairs are pushed towards agreement and impostor pairs
    below a margin.  Optimising Euclidean distance instead (the contrastive and
    triplet variants) only approximates this and leaves accuracy on the table
    once the codes are binarised.
    """
    L = T.shape[1]
    s = (T @ T.t()) / L
    same = (y[:, None] == y[None, :])
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos, neg = same & ~eye, ~same
    lp = (1.0 - s[pos]).mean() if pos.any() else T.sum() * 0.0
    ln = (F.relu(s[neg] - margin) ** 2).mean() if neg.any() else T.sum() * 0.0
    return lp + ln


def distill_loss(T, z_teacher):
    """Transfer the teacher's similarity structure into the protected space.

    The unprotected encoder reaches a lower EER than the binarised template can,
    so its pairwise geometry is a better training target than the labels alone.
    Matching the two similarity matrices - the teacher's cosine gram and the
    student's normalised soft-code inner product - pushes the protected space to
    reproduce that geometry under the constraint that it must survive
    binarisation.
    """
    L = T.shape[1]
    s_stu = (T @ T.t()) / L
    with torch.no_grad():
        z = F.normalize(z_teacher, dim=1)
        s_tea = z @ z.t()
    return F.mse_loss(s_stu, s_tea)


def template_loss(T, lam3=1.0):
    """Bit balance + bit decorrelation."""
    balance = (T.mean(0) ** 2).sum()
    L = T.shape[1]
    C = (T.t() @ T) / T.shape[0]
    I = torch.eye(L, device=T.device, dtype=T.dtype)
    decorr = ((C - I) ** 2).mean()
    return balance + lam3 * decorr


# ----------------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------------

def augment(xb, rng_state, p_scale=0.9, p_shift=0.9, p_noise=0.7,
            p_wander=0.5, shift_max=12):
    """Light, label-preserving ECG augmentation.

    Applied identically to every deep variant so the comparison stays fair.  The
    transforms mimic the nuisance factors the encoder must be invariant to:
    gain differences between acquisitions, R-peak localisation jitter, and
    sensor noise.
    """
    N, T = xb.shape
    dev = xb.device
    g = rng_state
    out = xb
    if p_scale > 0:
        s = torch.empty(N, 1, device=dev).uniform_(0.85, 1.15, generator=g)
        keep = (torch.rand(N, 1, device=dev, generator=g) < p_scale).float()
        out = out * (s * keep + (1 - keep))
    if p_shift > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (N,), device=dev,
                               generator=g)
        idx = (torch.arange(T, device=dev).unsqueeze(0) - shifts.unsqueeze(1)) % T
        moved = torch.gather(out, 1, idx)
        keep = (torch.rand(N, 1, device=dev, generator=g) < p_shift).float()
        out = moved * keep + out * (1 - keep)
    if p_noise > 0:
        n = torch.randn(N, T, device=dev, generator=g) * 0.06
        keep = (torch.rand(N, 1, device=dev, generator=g) < p_noise).float()
        out = out + n * keep
    if p_wander > 0:
        t = torch.linspace(0, 1, T, device=dev).unsqueeze(0)
        f = torch.empty(N, 1, device=dev).uniform_(0.5, 2.0, generator=g)
        ph = torch.empty(N, 1, device=dev).uniform_(0, 6.283, generator=g)
        amp = torch.empty(N, 1, device=dev).uniform_(0.0, 0.15, generator=g)
        keep = (torch.rand(N, 1, device=dev, generator=g) < p_wander).float()
        out = out + amp * keep * torch.sin(6.283 * f * t + ph)
    return out


class PKSampler:
    """P identities x K beats per batch, so the metric loss always sees pairs."""

    def __init__(self, labels, P=32, K=8, seed=0, steps=None):
        self.by = {}
        for i, y in enumerate(np.asarray(labels)):
            self.by.setdefault(int(y), []).append(i)
        self.ids = np.array([k for k, v in self.by.items() if len(v) >= 2])
        self.P, self.K = min(P, len(self.ids)), K
        self.rng = np.random.default_rng(seed)
        self.steps = steps or max(1, len(labels) // (self.P * self.K))

    def __iter__(self):
        for _ in range(self.steps):
            picks = self.rng.choice(self.ids, size=self.P, replace=False)
            idx = []
            for p in picks:
                pool = self.by[int(p)]
                take = self.rng.choice(len(pool), size=self.K,
                                       replace=len(pool) < self.K)
                idx += [pool[t] for t in take]
            yield np.asarray(idx)

    def __len__(self):
        return self.steps


# ----------------------------------------------------------------------------
# evaluation helper used for early stopping
# ----------------------------------------------------------------------------

@torch.no_grad()
def embed(model, X, device, bs=2048, hard=False):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(device)
        out.append(model.template(xb, hard=hard).cpu().numpy())
    return np.concatenate(out)


def verification_eer(model, data, m, split_key_e, split_key_q, device, protected):
    from metrics import eer_from_scores, hamming_similarity, cosine_similarity
    Xe, Xq = data["beats"][m[split_key_e]], data["beats"][m[split_key_q]]
    se, sq = data["subject"][m[split_key_e]], data["subject"][m[split_key_q]]
    if len(Xe) == 0 or len(Xq) == 0:
        return float("nan"), float("nan")
    Te = embed(model, Xe, device, hard=protected)
    Tq = embed(model, Xq, device, hard=protected)
    if protected:
        S = hamming_similarity(Tq.astype(np.uint8), Te.astype(np.uint8))
    else:
        S = cosine_similarity(Tq, Te)
    sc, lb, _, _ = protocols.aggregate_scores(S, se, sq)
    return eer_from_scores(sc, lb)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--proc", default=f"{_ROOT}/data/proc")
    ap.add_argument("--tag", default="",
                    help="run directory name; defaults to dataset_variant_L_s")
    ap.add_argument("--out", default=f"{_ROOT}/runs")
    ap.add_argument("--variant", default="proposed", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--gamma_start", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--l1", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--l3", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--app_key", default="app-A")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="cap optimisation steps per epoch (0 = one full pass); "
                         "used to keep the hyper-parameter search affordable on "
                         "the larger datasets")
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--teacher", default="", help="checkpoint of a trained "
                    "unprotected encoder used for similarity distillation")
    ap.add_argument("--l4", type=float, default=0.0, help="distillation weight")
    ap.add_argument("--arch", default="res", choices=["res", "plain"])
    ap.add_argument("--head", default="arc", choices=["arc", "linear"])
    ap.add_argument("--arc_s", type=float, default=30.0)
    ap.add_argument("--arc_m", type=float, default=0.30)
    ap.add_argument("--metric_kind", default="hamming",
                    choices=["triplet", "contrastive", "hamming"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--val_smooth", type=int, default=3)
    a = ap.parse_args()

    cfg = VARIANTS[a.variant]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = protocols.load(os.path.join(a.proc, f"{a.dataset}.npz"))
    split = protocols.make_split(data, a.dataset, seed=a.seed)
    m = protocols.masks(data, split)
    stats = protocols.split_stats(data, split)

    Xtr = data["beats"][m["train"]]
    ytr_raw = data["subject"][m["train"]]
    classes = np.unique(ytr_raw)
    remap = {int(c): i for i, c in enumerate(classes)}
    ytr = np.array([remap[int(v)] for v in ytr_raw], dtype=np.int64)

    tag = a.tag or f"{a.dataset}_{a.variant}_L{a.L}_s{a.seed}"
    run_dir = os.path.join(a.out, tag)
    os.makedirs(run_dir, exist_ok=True)
    protocols.save_split(split, os.path.join(run_dir, "split.json"))
    with open(os.path.join(run_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{tag}] device={device}  train beats={len(Xtr)}  classes={len(classes)}",
          flush=True)
    print(f"[{tag}] split stats: " +
          json.dumps({k: v for k, v in stats.items() if isinstance(v, dict)}),
          flush=True)

    # the fixed-projection ablation reuses one seed for every application key
    model = ProtectedNet(
        n_classes=len(classes), d=a.d, L=a.L, gamma=a.gamma,
        app_key=(a.app_key if cfg["keyed"] else "FIXED-PROJECTION"),
        use_cancelable=cfg["cancel"], arch=a.arch, head=a.head,
        arc_s=a.arc_s, arc_m=a.arc_m, drop=a.drop,
    ).to(device)

    teacher = None
    if a.teacher and a.l4 > 0:
        from evaluate import load_model as _load
        teacher, _, _ = _load(a.teacher, device)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"[{tag}] distilling from {a.teacher}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    def lr_at(ep):
        if ep < a.warmup:
            return (ep + 1) / max(1, a.warmup)
        p = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    sampler = PKSampler(ytr, P=a.P, K=a.K, seed=a.seed,
                        steps=(a.steps_per_epoch or None))
    gen = torch.Generator(device=device); gen.manual_seed(a.seed + 12345)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr)

    best, best_ep, bad = float("inf"), -1, 0
    history = []
    t0 = time.time()

    for ep in range(a.epochs):
        model.train()
        # Anneal the quantiser steepness.  tanh(gamma*u) with u rescaled to unit
        # variance saturates for gamma >~ 2, which kills the gradient through the
        # binarisation.  Starting soft and sharpening keeps gradients alive early
        # while still producing a crisp sign decision by the end of training.
        if model.cancel is not None and a.epochs > 1:
            frac = ep / (a.epochs - 1)
            model.cancel.gamma = float(
                a.gamma_start * (a.gamma / a.gamma_start) ** frac)
        agg = dict(loss=0.0, cls=0.0, met=0.0, tmp=0.0, n=0)
        for idx in sampler:
            xb = Xtr_t[idx].to(device, non_blocking=True)
            yb = ytr_t[idx].to(device, non_blocking=True)
            if a.augment:
                xb = augment(xb, gen)
            T, logits = model(xb, yb)

            loss = torch.zeros((), device=device)
            lc = lm = lt = torch.zeros((), device=device)
            if cfg["cls"]:
                lc = F.cross_entropy(logits, yb)
                loss = loss + lc
            if cfg["metric"]:
                if a.metric_kind == "hamming":
                    lm = hamming_metric_loss(T, yb, margin=a.margin)
                elif a.metric_kind == "triplet":
                    lm = triplet_loss(T, yb, margin=a.margin)
                else:
                    lm = metric_loss(T, yb, margin=a.margin)
                loss = loss + a.l1 * lm
            if cfg["temp"]:
                lt = template_loss(T, lam3=a.l3)
                loss = loss + a.l2 * lt
            if teacher is not None and model.cancel is not None:
                with torch.no_grad():
                    zt = teacher.encoder(xb)
                loss = loss + a.l4 * distill_loss(T, zt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            agg["loss"] += float(loss.detach()) ; agg["cls"] += float(lc.detach())
            agg["met"] += float(lm.detach()) ; agg["tmp"] += float(lt.detach())
            agg["n"] += 1
        sched.step()

        protected = cfg["cancel"]
        veer, _ = verification_eer(model, data, m, "val_enrol", "val_query",
                                   device, protected)
        n = max(agg["n"], 1)
        rec = dict(epoch=ep, loss=agg["loss"] / n, cls=agg["cls"] / n,
                   metric=agg["met"] / n, temp=agg["tmp"] / n, val_eer=veer)
        history.append(rec)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} loss={rec['loss']:.4f} cls={rec['cls']:.4f} "
                  f"met={rec['metric']:.4f} tmp={rec['temp']:.4f} "
                  f"val_eer={veer:.3f}%", flush=True)

        recent = [h["val_eer"] for h in history[-a.val_smooth:]
                  if np.isfinite(h["val_eer"])]
        veer_s = float(np.mean(recent)) if recent else veer
        rec["val_eer_smooth"] = veer_s
        if np.isfinite(veer_s) and veer_s < best - 1e-4:
            best, best_ep, bad = veer_s, ep, 0
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "classes": classes.tolist(), "val_eer": veer, "epoch": ep,
                        "gamma_at_best": (model.cancel.gamma
                                          if model.cancel is not None else None)},
                       os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at ep{ep} (best ep{best_ep} val_eer={best:.3f})",
                      flush=True)
                break

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(dict(history=history, best_val_eer=best, best_epoch=best_ep,
                       minutes=(time.time() - t0) / 60.0, variant=a.variant,
                       config=cfg), f, indent=2)
    print(f"[{tag}] done  best_val_eer={best:.3f}%  "
          f"({(time.time() - t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
