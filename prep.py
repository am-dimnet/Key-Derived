"""
Preprocessing pipeline for deep cancelable ECG template learning.

Produces, for each dataset, a single .npz holding:
    beats      float32 [N, 512]   amplitude-normalised single-heartbeat segments
    subject    int32   [N]        subject index (0..S-1)
    record     int32   [N]        record index, unique within dataset
    session    int32   [N]        session index (dataset-specific; 0 if not applicable)
    rpeak      int64   [N]        R-peak sample index inside the source record
plus a JSON sidecar with the QC ledger required by reviewer comment R1.8.

Design decisions that answer specific reviewer comments:
  R1.9  every dataset is resampled to a common FS_TARGET before R-peak detection
        and segmentation, so the fixed 512-sample window spans the same 1024 ms
        everywhere.
  R1.8  every record and every candidate beat that is dropped is counted against
        a named reason, and the ledger is written out alongside the beats.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, find_peaks

from paths import ROOT as _ROOT

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

FS_TARGET = 500          # Hz, common rate for all datasets (R1.9)
SEG_LEN = 512            # samples -> 1024 ms at FS_TARGET
PRE = 200                # samples before R peak (400 ms, covers P wave)
POST = SEG_LEN - PRE     # 312 samples after R peak (624 ms, covers T wave)

BP_LOW, BP_HIGH = 0.5, 40.0   # band-pass for the analysis signal
RR_MIN, RR_MAX = 0.4, 2.0     # s, physiological plausibility (150..30 bpm)
RR_DEV = 0.30                 # max relative deviation from local median RR
FLAT_EPS = 1e-6               # amplitude below which a beat counts as flat
MAX_ABS_Z = 12.0              # crude artefact rejection on the normalised beat


# ----------------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------------

def bandpass(x, fs, lo=BP_LOW, hi=BP_HIGH, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.98)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def to_target_fs(x, fs_in, fs_out=FS_TARGET):
    """Polyphase resampling to the common analysis rate (R1.9)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(np.asarray(x, dtype=np.float64), fs_out // g, fs_in // g)


def despike(x, k=10.0, p=99.0):
    """Interpolate over gross amplitude artefacts.

    The reference scale is the p-th percentile of |x - median|, which for an ECG
    is about the R-peak amplitude, so a threshold of k times that removes the
    isolated excursions seen in finger recordings (three orders of magnitude
    above the ECG) without touching genuine QRS complexes.  A MAD-based clip is
    deliberately not used: in recordings with long constant stretches the MAD
    collapses to zero and the clip would flatten the whole signal.
    """
    x = np.asarray(x, float)
    med = np.median(x)
    scale = np.percentile(np.abs(x - med), p) + 1e-12
    bad = np.abs(x - med) > k * scale
    if bad.any() and (~bad).sum() > 10:
        idx = np.flatnonzero(~bad)
        x = np.interp(np.arange(len(x)), idx, x[idx])
    return x


def _mwi(x, fs):
    """Pan-Tompkins energy envelope: 5-15 Hz band, derivative, square, integrate."""
    nyq = fs / 2.0
    b, a = butter(2, [5.0 / nyq, min(15.0, nyq * 0.98) / nyq], btype="band")
    y = filtfilt(b, a, x)
    y = np.diff(y, prepend=y[0]) ** 2
    win = max(1, int(0.150 * fs))
    return np.convolve(y, np.ones(win) / win, mode="same")


def _dominant_rr(env, fs, lo=0.33, hi=2.0, max_seconds=30.0):
    """Dominant cardiac period from the autocorrelation of the energy envelope.

    The envelope has one dominant lobe per QRS, so its autocorrelation peaks at
    the true RR interval even when individual complexes are noisy.  This is what
    stops the detector from locking onto R+T pairs and reporting double the true
    heart rate.

    The autocorrelation is computed by FFT and on at most `max_seconds` of
    signal: a direct O(n^2) correlation over a multi-minute clinical recording
    costs billions of operations per record for no extra accuracy, since half a
    minute already contains dozens of beats.
    """
    z = env[: int(max_seconds * fs)]
    z = z - z.mean()
    s = z.std()
    if s < 1e-12 or len(z) < int(4 * fs):
        return None
    z = z / s
    n = 1 << int(np.ceil(np.log2(2 * len(z))))
    F = np.fft.rfft(z, n)
    ac = np.fft.irfft(F * np.conj(F), n)[: len(z)]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    a, b = int(lo * fs), min(int(hi * fs), len(ac) - 1)
    if b <= a:
        return None
    seg = ac[a:b]
    pk, _ = find_peaks(seg)
    if len(pk) == 0:
        return None
    h = seg[pk]
    if h.max() <= 0.05:
        return None
    # The autocorrelation also peaks at 2RR, 3RR, ...  Take the FIRST peak that
    # reaches half the strongest one, so the fundamental is picked rather than a
    # harmonic - otherwise the refractory period doubles and half the beats are
    # silently dropped.
    first = int(pk[int(np.argmax(h >= 0.5 * h.max()))])
    return (a + first) / fs


def detect_rpeaks(x, fs=FS_TARGET):
    """Pan-Tompkins with an autocorrelation-derived refractory period and
    slope-based T-wave rejection."""
    env = _mwi(x, fs)
    if not np.isfinite(env).all() or env.max() <= 0:
        return np.array([], dtype=np.int64)

    rr0 = _dominant_rr(env, fs)
    # refractory period: 60% of the estimated RR, clamped to a sane band
    min_rr = 0.28 if rr0 is None else float(np.clip(0.60 * rr0, 0.28, 1.5))

    thr = 0.30 * np.mean(env[env > np.percentile(env, 75)])
    peaks, _ = find_peaks(env, height=max(thr, 1e-12), distance=int(min_rr * fs))
    # If the peak train is still about twice as dense as the autocorrelation
    # period implies, the detector is firing on R and T; tighten and retry once.
    if rr0 is not None and len(peaks) > 2:
        if float(np.median(np.diff(peaks))) / fs < 0.65 * rr0:
            peaks, _ = find_peaks(env, height=max(thr, 1e-12),
                                  distance=int(0.85 * rr0 * fs))
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)

    # ---- refine onto the true fiducial point, with a single polarity per record
    w = int(0.05 * fs)
    cand = []
    for p in peaks:
        lo, hi = max(0, p - w), min(len(x), p + w + 1)
        if hi > lo:
            seg = x[lo:hi]
            cand.append((lo + int(np.argmax(seg)), lo + int(np.argmin(seg))))
    if not cand:
        return np.array([], dtype=np.int64)
    pos = np.array([c[0] for c in cand])
    neg = np.array([c[1] for c in cand])
    # choose the polarity whose extrema are more consistent in amplitude
    ap, an = np.abs(x[pos]), np.abs(x[neg])
    use_pos = (np.median(ap) / (ap.std() + 1e-9)) >= (np.median(an) / (an.std() + 1e-9))
    r = np.unique(pos if use_pos else neg)

    # ---- Pan-Tompkins T-wave discrimination on the surviving peaks
    if len(r) > 2:
        slope = np.abs(np.gradient(x))
        keep = [r[0]]
        for p in r[1:]:
            gap = (p - keep[-1]) / fs
            if gap < 0.36 and slope[p] < 0.5 * slope[keep[-1]]:
                continue                      # T wave, not a QRS
            if gap < 0.20:
                continue                      # inside the absolute refractory period
            keep.append(p)
        r = np.asarray(keep, dtype=np.int64)
    return np.asarray(r, dtype=np.int64)


def segment_beats(x, rpeaks, fs=FS_TARGET):
    """Cut fixed-length beats and apply the QC rules. Returns beats, keep_idx, reasons."""
    reasons = Counter()
    if len(rpeaks) < 3:
        reasons["record_too_few_rpeaks"] += 1
        return np.zeros((0, SEG_LEN), np.float32), np.zeros((0,), np.int64), reasons

    rr = np.diff(rpeaks) / fs
    med_rr = float(np.median(rr))
    if not (RR_MIN <= med_rr <= RR_MAX):
        reasons["record_implausible_median_rr"] += 1
        return np.zeros((0, SEG_LEN), np.float32), np.zeros((0,), np.int64), reasons

    beats, kept = [], []
    for i, r in enumerate(rpeaks):
        # RR plausibility: both neighbouring intervals must be sane
        prev_rr = (r - rpeaks[i - 1]) / fs if i > 0 else med_rr
        next_rr = (rpeaks[i + 1] - r) / fs if i + 1 < len(rpeaks) else med_rr
        if not (RR_MIN <= prev_rr <= RR_MAX and RR_MIN <= next_rr <= RR_MAX):
            reasons["beat_rr_out_of_range"] += 1
            continue
        if abs(prev_rr - med_rr) / med_rr > RR_DEV or abs(next_rr - med_rr) / med_rr > RR_DEV:
            reasons["beat_rr_deviates_from_local_median"] += 1
            continue
        lo, hi = r - PRE, r + POST
        if lo < 0 or hi > len(x):
            reasons["beat_window_out_of_record"] += 1
            continue
        seg = x[lo:hi].astype(np.float64)
        rng = seg.max() - seg.min()
        if not np.isfinite(seg).all():
            reasons["beat_non_finite"] += 1
            continue
        if rng < FLAT_EPS:
            reasons["beat_flat"] += 1
            continue
        # amplitude normalisation: zero mean, unit std, then artefact check
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        if np.abs(seg).max() > MAX_ABS_Z:
            reasons["beat_amplitude_artefact"] += 1
            continue
        beats.append(seg.astype(np.float32))
        kept.append(int(r))

    if not beats:
        return np.zeros((0, SEG_LEN), np.float32), np.zeros((0,), np.int64), reasons
    return np.stack(beats), np.asarray(kept, dtype=np.int64), reasons


def process_record(sig, fs_in):
    """Full per-record chain. Returns (beats, rpeaks_at_target_fs, reasons)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    if x.size < 3 * fs_in:
        return None, None, Counter({"record_too_short": 1})
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = to_target_fs(x, fs_in, FS_TARGET)
    if x.size < SEG_LEN + 2 * FS_TARGET:
        return None, None, Counter({"record_too_short_after_resample": 1})
    if float(np.std(x)) < FLAT_EPS:
        return None, None, Counter({"record_flat": 1})
    x = bandpass(despike(x), FS_TARGET)
    rp = detect_rpeaks(x, FS_TARGET)
    beats, kept, reasons = segment_beats(x, rp, FS_TARGET)
    return beats, kept, reasons


# ----------------------------------------------------------------------------
# dataset readers -> list of dicts(subject, record_name, session, signal, fs)
# ----------------------------------------------------------------------------

def read_ecgid(root):
    """ECG-ID: Person_XX/rec_N. Channel 0 is the raw ECG lead I. 500 Hz.
    Session index = recording ordinal, which is the paper's enrol/test axis."""
    import wfdb
    items = []
    persons = sorted(d for d in os.listdir(root) if d.startswith("Person_"))
    for pid, person in enumerate(persons):
        pdir = os.path.join(root, person)
        recs = sorted(
            (f[:-4] for f in os.listdir(pdir) if f.endswith(".hea")),
            key=lambda s: int(re.search(r"\d+", s).group()),
        )
        for ridx, rec in enumerate(recs):
            try:
                r = wfdb.rdrecord(os.path.join(pdir, rec))
            except Exception:
                continue
            meta = {"age": None, "sex": None}
            for c in (r.comments or []):
                m = re.search(r"Age:\s*(\d+)", c)
                if m:
                    meta["age"] = int(m.group(1))
                m = re.search(r"Sex:\s*(\w+)", c)
                if m:
                    meta["sex"] = m.group(1).strip().lower()
            items.append(dict(subject=person, record=f"{person}/{rec}", session=ridx,
                              signal=r.p_signal[:, 0], fs=int(r.fs), meta=meta))
    return items


HP_FS = 250          # true rate: the live part of every record is 3750 samples / 15 s


def _clean_heartprint(sig):
    """Strip the trailing zero padding and the per-beat zero markers.

    Every exported record is 7500 lines long, but only the leading ~3750 samples
    carry data (15 s at 250 Hz); the remainder is zero padding.  The acquisition
    software also writes a single 0.0 immediately before each QRS, which shows up
    as an isolated dropout roughly once per beat.  Both are removed here, so the
    signal handed to the detector is the genuine 250 Hz waveform.
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    nz = x != 0
    if not nz.any():
        return None
    # cut the long trailing run of zeros
    last = int(np.flatnonzero(nz)[-1])
    x = x[: last + 1]
    # interpolate the isolated zero markers that remain inside the live part
    bad = x == 0
    if bad.any() and (~bad).sum() > 10:
        idx = np.flatnonzero(~bad)
        x = np.interp(np.arange(len(x)), idx, x[idx])
    return x


def read_heartprint(root):
    """Heartprint: Session-*/subject/timestamp_ECG.txt, single column, 250 Hz."""
    items = []
    sess_order = {"Session-1": 0, "Session-2": 1, "Session-3L": 2, "Session-3R": 3}
    base = os.path.join(root, "Heartprint")
    if not os.path.isdir(base):
        base = root
    for sess in sorted(os.listdir(base)):
        sdir = os.path.join(base, sess)
        if not os.path.isdir(sdir):
            continue
        sidx = sess_order.get(sess, len(sess_order))
        for subj in sorted(os.listdir(sdir)):
            udir = os.path.join(sdir, subj)
            if not os.path.isdir(udir):
                continue
            for f in sorted(os.listdir(udir)):
                if not f.endswith(".txt"):
                    continue
                try:
                    sig = _clean_heartprint(np.loadtxt(os.path.join(udir, f)))
                except Exception:
                    continue
                if sig is None or len(sig) < 4 * HP_FS:
                    continue
                items.append(dict(subject=f"HP_{subj}", record=f"{sess}/{subj}/{f[:-4]}",
                                  session=sidx, signal=sig, fs=HP_FS, meta={}))
    return items


def read_ptbdb(root):
    """PTB Diagnostic: patientXXX/sYYYY. Lead i is the unified input lead. 1000 Hz.

    Each record splits its 15 signals over two files: the 12 conventional leads
    live in the .dat and the three Frank leads in a .xyz.  Reading the record
    whole therefore fails whenever the .xyz is absent, so lead I is requested by
    name and only the .dat is touched.
    """
    import wfdb
    items = []
    heas = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(root) for f in fs if f.endswith(".hea")
    )
    for hea in heas:
        rec = hea[:-4]
        if not os.path.exists(rec + ".dat"):
            continue
        try:
            r = wfdb.rdrecord(rec, channel_names=["i"])
        except Exception:
            continue
        names = [str(n).strip().lower() for n in (r.sig_name or [])]
        if "i" not in names:
            continue
        ch = names.index("i")
        patient = os.path.basename(os.path.dirname(rec))
        dx = None
        for c in (r.comments or []):
            m = re.search(r"Reason for admission:\s*(.+)", c, re.I)
            if m:
                dx = m.group(1).strip().lower()
        items.append(dict(subject=patient, record=f"{patient}/{os.path.basename(rec)}",
                          session=0, signal=r.p_signal[:, ch], fs=int(r.fs),
                          meta={"diagnosis": dx}))
    return items


READERS = {"ecgid": read_ecgid, "heartprint": read_heartprint, "ptbdb": read_ptbdb}


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def _worker(args):
    idx, sig, fs = args
    beats, kept, reasons = process_record(sig, fs)
    return idx, beats, kept, dict(reasons)


def build(dataset, raw_root, out_dir, min_beats=5, min_records=2, workers=16):
    print(f"[{dataset}] reading raw records from {raw_root} ...", flush=True)
    items = READERS[dataset](raw_root)
    print(f"[{dataset}] {len(items)} raw records, "
          f"{len(set(i['subject'] for i in items))} subjects", flush=True)

    ledger = Counter()
    ledger["records_read"] = len(items)

    payload = [(i, it["signal"], it["fs"]) for i, it in enumerate(items)]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, beats, kept, reasons in ex.map(_worker, payload, chunksize=4):
            results[idx] = (beats, kept)
            for k, v in reasons.items():
                ledger[k] += v

    # ---- assemble, dropping records then subjects that fail the minimum counts
    per_subject = defaultdict(list)
    for idx, it in enumerate(items):
        beats, kept = results.get(idx, (None, None))
        if beats is None or len(beats) < min_beats:
            ledger["records_dropped_too_few_valid_beats"] += 1
            continue
        per_subject[it["subject"]].append((idx, beats, kept))

    subjects = []
    for s, recs in per_subject.items():
        if len({items[i]["record"] for i, _, _ in recs}) < min_records:
            ledger["subjects_dropped_too_few_records"] += 1
            ledger["records_dropped_with_those_subjects"] += len(recs)
            continue
        subjects.append(s)
    subjects = sorted(subjects)
    sub2idx = {s: i for i, s in enumerate(subjects)}

    B, S, R, SE, RP = [], [], [], [], []
    rec_names, rec_meta = [], []
    for s in subjects:
        for idx, beats, kept in sorted(per_subject[s], key=lambda t: items[t[0]]["record"]):
            it = items[idx]
            rid = len(rec_names)
            rec_names.append(it["record"])
            rec_meta.append(it.get("meta", {}))
            B.append(beats)
            S.append(np.full(len(beats), sub2idx[s], np.int32))
            R.append(np.full(len(beats), rid, np.int32))
            SE.append(np.full(len(beats), it["session"], np.int32))
            RP.append(kept)

    beats = np.concatenate(B).astype(np.float32)
    subject = np.concatenate(S)
    record = np.concatenate(R)
    session = np.concatenate(SE)
    rpeak = np.concatenate(RP)

    ledger["subjects_retained"] = len(subjects)
    ledger["records_retained"] = len(rec_names)
    ledger["beats_retained"] = int(len(beats))
    ledger["beats_discarded"] = int(sum(
        v for k, v in ledger.items() if k.startswith("beat_")))

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{dataset}.npz"),
        beats=beats, subject=subject, record=record, session=session, rpeak=rpeak,
        subjects=np.array(subjects, dtype=object),
        record_names=np.array(rec_names, dtype=object),
    )
    meta = dict(
        dataset=dataset, fs_target=FS_TARGET, seg_len=SEG_LEN, pre=PRE, post=POST,
        window_ms=1000.0 * SEG_LEN / FS_TARGET,
        bandpass=[BP_LOW, BP_HIGH], rr_range=[RR_MIN, RR_MAX], rr_dev=RR_DEV,
        min_beats_per_record=min_beats, min_records_per_subject=min_records,
        ledger=dict(sorted(ledger.items())),
        record_names=rec_names, record_meta=rec_meta, subjects=subjects,
    )
    with open(os.path.join(out_dir, f"{dataset}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{dataset}] retained {len(subjects)} subjects / {len(rec_names)} records "
          f"/ {len(beats)} beats", flush=True)
    for k, v in sorted(ledger.items()):
        print(f"    {k:45s} {v}", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(READERS))
    ap.add_argument("--raw", default=None,
                    help="dataset directory; defaults to $ARTICLE4_ROOT/data/raw/<dataset>")
    ap.add_argument("--out", default=f"{_ROOT}/data/proc")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.raw is None:
        a.raw = os.path.join(_ROOT, "data", "raw", a.dataset)
    build(a.dataset, a.raw, a.out, workers=a.workers)
