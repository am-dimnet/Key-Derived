"""
Encoder, key-derived cancelable template layer, and the protection baselines.

The key schedule is deliberately more general than the submitted manuscript: a
protected template is derived from the triple (application, user, version), and
each component can be switched off.  That lets the same code reproduce the
application-only setting described in the paper and run the version-rollover
experiments that reviewer comment R1.7 asks for.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# key schedule  (R1.4: entropy, derivation, versioning)
# ----------------------------------------------------------------------------

KEY_BITS = 256


def derive_seed(app_key: str, user: str = "", version: int = 0,
                bind_user: bool = True, bind_version: bool = True) -> int:
    """SHA-256 over the key material; the low 63 bits seed the CSPRNG."""
    h = hashlib.sha256()
    h.update(app_key.encode())
    if bind_user:
        h.update(b"|u|" + str(user).encode())
    if bind_version:
        h.update(b"|v|" + str(version).encode())
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


def make_transform(d: int, L: int, seed: int):
    """Orthogonal random projection R (L x d) and bias b (L,) from one seed.

    A Gaussian d x d matrix is QR-decomposed and the first L orthonormal rows are
    kept, so R R^T = I_L exactly.  The bias is drawn from U(-rho, rho) with
    rho = 1/sqrt(d), giving key-dependent threshold offsets.
    """
    assert L <= d, f"template length L={L} cannot exceed embedding dim d={d}"
    g = np.random.default_rng(seed)
    G = g.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(R))          # fix QR sign ambiguity -> deterministic
    Rk = Q.T[:L].copy()
    rho = 1.0 / np.sqrt(d)
    bk = g.uniform(-rho, rho, size=L)
    return Rk.astype(np.float32), bk.astype(np.float32)


# ----------------------------------------------------------------------------
# encoder
# ----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=7, pool=2, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.MaxPool1d(pool), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class SE(nn.Module):
    """Squeeze-and-excitation over channels."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(4, c // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(4, c // r), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(-1)).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Pre-activation residual block with SE, then decimation."""

    def __init__(self, cin, cout, k=7, pool=2, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
        )
        self.se = SE(cout)
        self.skip = (nn.Identity() if cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False),
                                   nn.BatchNorm1d(cout)))
        self.post = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool1d(pool),
                                  nn.Dropout(drop))

    def forward(self, x):
        return self.post(self.se(self.conv(x)) + self.skip(x))


class GeM(nn.Module):
    """Generalised-mean pooling; interpolates between average and max pooling."""

    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return (x.clamp(min=1e-6).pow(p).mean(-1)).pow(1.0 / p)


class ECGEncoder(nn.Module):
    """1D-CNN over a single 512-sample heartbeat -> d-dimensional embedding.

    `arch="res"` is the wider residual/SE backbone used for the final results;
    `arch="plain"` is the original stacked-convolution encoder, kept so the
    architecture change can itself be ablated.
    """

    def __init__(self, d=512, widths=(64, 128, 256, 512), drop=0.1, arch="res"):
        super().__init__()
        self.arch = arch
        if arch == "plain":
            blocks, cin = [], 1
            for w in widths:
                blocks.append(ConvBlock(cin, w, drop=drop))
                cin = w
            self.body = nn.Sequential(*blocks)
            self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten())
            feat = cin * 4
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(1, widths[0], 15, padding=7, bias=False),
                nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
            blocks, cin = [], widths[0]
            for w in widths:
                blocks.append(ResBlock(cin, w, drop=drop))
                cin = w
            self.body = nn.Sequential(*blocks)
            self.pool = GeM()
            feat = cin
        self.head = nn.Sequential(nn.Linear(feat, d), nn.BatchNorm1d(d))
        self.d = d

    def forward(self, x):                      # x: [N, 512]
        h = x.unsqueeze(1)
        if self.arch != "plain":
            h = self.stem(h)
        z = self.head(self.pool(self.body(h)))
        return F.normalize(z, dim=1)           # unit sphere (Eq. 2)


class ArcMargin(nn.Module):
    """Additive angular margin head (ArcFace).

    Replaces the plain linear classifier in L_cls.  The margin is applied in
    angular space, which is a much stronger identity-discrimination signal than
    plain softmax and is standard in biometric recognition.  It is available to
    every variant, so the comparison stays a statement about protected-space
    learning rather than about classifier choice.
    """

    def __init__(self, in_dim, n_classes, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, x, y=None):
        cos = F.linear(F.normalize(x, dim=1), F.normalize(self.W, dim=1)).clamp(-1 + 1e-7,
                                                                                1 - 1e-7)
        if y is None:
            return self.s * cos
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, self.W.shape[0]).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


# ----------------------------------------------------------------------------
# cancelable template layer
# ----------------------------------------------------------------------------

class CancelableLayer(nn.Module):
    """u = R_k z + b_k ;  soft T = tanh(gamma u) ;  hard T = 1[T_soft >= 0].

    R_k and b_k are buffers, never updated by gradient descent, exactly as in the
    manuscript.  `rekey` swaps them in place so a whole revocation round costs a
    forward pass rather than a retrain.
    """

    def __init__(self, d=512, L=256, gamma=2.0, app_key="app-A", version=0,
                 bind_user=False, bind_version=True):
        super().__init__()
        self.d, self.L, self.gamma = d, L, gamma
        self.bind_user, self.bind_version = bind_user, bind_version
        self.register_buffer("R", torch.zeros(L, d))
        self.register_buffer("b", torch.zeros(L))
        self.rekey(app_key, version=version)

    @torch.no_grad()
    def rekey(self, app_key, user="", version=0):
        seed = derive_seed(app_key, user, version,
                           self.bind_user, self.bind_version)
        Rk, bk = make_transform(self.d, self.L, seed)
        self.R.copy_(torch.from_numpy(Rk).to(self.R.device))
        self.b.copy_(torch.from_numpy(bk).to(self.b.device))
        self.app_key, self.user, self.version = app_key, user, version
        return self

    def forward(self, z, hard=False):
        # R has orthonormal rows and z is unit norm, so each component of Rz is
        # O(1/sqrt(d)) and the bias is drawn on the same scale.  Without the
        # sqrt(d) rescaling, tanh(gamma*u) would sit at about +-0.2 for any
        # sensible gamma: the soft template never saturates, the sign that
        # defines the bit is decided by a vanishing margin, and the
        # decorrelation term (which pulls the diagonal of T^T T / N towards 1)
        # ends up fighting the rest of the objective.  Rescaling to unit
        # variance makes gamma mean what the manuscript intends.
        u = F.linear(z, self.R, self.b) * (self.d ** 0.5)
        t = torch.tanh(self.gamma * u)
        if hard:
            return (t >= 0).float()
        return t


class ProtectedNet(nn.Module):
    """Encoder + cancelable layer + a classifier head used only during training."""

    def __init__(self, n_classes, d=512, L=256, gamma=2.0, app_key="app-A",
                 version=0, bind_user=False, use_cancelable=True, drop=0.1,
                 arch="res", head="arc", arc_s=30.0, arc_m=0.30):
        super().__init__()
        self.encoder = ECGEncoder(d=d, drop=drop, arch=arch)
        self.use_cancelable = use_cancelable
        self.cancel = (CancelableLayer(d, L, gamma, app_key, version, bind_user)
                       if use_cancelable else None)
        out_dim = L if use_cancelable else d
        self.head_kind = head
        self.classifier = (ArcMargin(out_dim, n_classes, arc_s, arc_m)
                           if head == "arc" else nn.Linear(out_dim, n_classes))
        self.out_dim = out_dim

    def template(self, x, hard=False):
        z = self.encoder(x)
        if self.cancel is None:
            return z                            # unprotected embedding baseline
        return self.cancel(z, hard=hard)

    def forward(self, x, y=None):
        t = self.template(x, hard=False)
        logits = (self.classifier(t, y) if self.head_kind == "arc"
                  else self.classifier(t))
        return t, logits


# ----------------------------------------------------------------------------
# post-hoc protection baselines
# ----------------------------------------------------------------------------

class BioHashing:
    """Classic BioHashing: key-seeded Gaussian projection, Gram-Schmidt, sign.

    Applied to a frozen embedding, i.e. the sequential/post-hoc pipeline that the
    manuscript compares against.
    """

    def __init__(self, d, L, key="app-A", user="", version=0, bind_user=False):
        seed = derive_seed(key, user, version, bind_user, True)
        g = np.random.default_rng(seed)
        M = g.standard_normal((d, L))
        Q, _ = np.linalg.qr(M)                  # orthonormal columns
        self.W = Q[:, :L].astype(np.float32)

    def __call__(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        P = Z @ self.W
        thr = np.median(P, axis=0, keepdims=True)   # standard BioHash threshold
        return (P >= thr).astype(np.uint8)


class PostHocCancelable:
    """Our key-derived transform applied to a frozen embedding.

    This is the single-variable control demanded by reviewer comment R1.1: the
    transform, template length and matcher are identical to the proposed system,
    and only the joint optimisation is removed.
    """

    def __init__(self, d, L, key="app-A", user="", version=0,
                 bind_user=False, gamma=2.0):
        seed = derive_seed(key, user, version, bind_user, True)
        self.R, self.b = make_transform(d, L, seed)
        self.gamma = gamma

    def __call__(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        U = (Z @ self.R.T + self.b) * (self.R.shape[1] ** 0.5)
        return (np.tanh(self.gamma * U) >= 0).astype(np.uint8)


class RandomProjection:
    """Cancelable template by key-seeded Gaussian random projection and sign.

    The classical random-projection scheme: unlike BioHashing the projection is
    not orthonormalised and the threshold is zero rather than the per-dimension
    median, so the two differ in both the geometry of the projection and the
    quantiser.
    """

    def __init__(self, d, L, key="app-A", user="", version=0, bind_user=False):
        seed = derive_seed(key, user, version, bind_user, True)
        g = np.random.default_rng(seed)
        self.W = (g.standard_normal((d, L)) / np.sqrt(d)).astype(np.float32)

    def __call__(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        return ((Z @ self.W) >= 0).astype(np.uint8)


class CompressiveSensing:
    """Cancelable template by key-seeded compressive measurement and 1-bit
    quantisation.

    A sparse Bernoulli measurement matrix with L < d rows is drawn from the key,
    the embedding is measured, and the sign of each measurement is stored.  This
    is the one-bit compressive-sensing form used for cancelable biometrics: the
    measurement is structured rather than dense Gaussian, and the compression
    ratio itself is part of the protection.
    """

    def __init__(self, d, L, key="app-A", user="", version=0, bind_user=False,
                 density=0.25):
        seed = derive_seed(key, user, version, bind_user, True)
        g = np.random.default_rng(seed)
        mask = g.random((d, L)) < density
        signs = g.choice(np.array([-1.0, 1.0]), size=(d, L))
        self.Phi = (mask * signs / np.sqrt(density * d)).astype(np.float32)

    def __call__(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        Y = Z @ self.Phi
        return ((Y - np.median(Y, axis=0, keepdims=True)) >= 0).astype(np.uint8)
