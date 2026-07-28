"""Per-subset RECORD counts and Heartprint inter-session intervals.

Two reviewer requests are reporting gaps rather than missing experiments:

  R1.8 asks for subjects, *records* and heartbeat segments per split.  Table II
       currently reports subjects and beats; the record counts exist in the
       split statistics but were never carried into the paper.  They are also
       collected here across all five seeds, not seed 0 alone, since the subject
       partition differs per seed.

  R1.10 asks for the distribution of elapsed time between the enrolment session
       and the test sessions on Heartprint.  The manuscript says only "months to
       years".  Heartprint ships per-record session labels; where acquisition
       dates are available we report the interval distribution directly, and
       where they are not we say so rather than inventing a figure.

Writes results/tables/protocol_counts.json.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

import protocols

from paths import ROOT as _ROOT

R = f"{_ROOT}"
SUBSETS = ("train", "val_enrol", "val_query", "enrol", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--proc", default=f"{R}/data/proc")
    ap.add_argument("--out", default=f"{R}/results/tables/protocol_counts.json")
    a = ap.parse_args()

    out = {}
    for ds in a.datasets.split(","):
        data = protocols.load(os.path.join(a.proc, f"{ds}.npz"))
        rec = data["record"]
        subj = data["subject"]
        per_seed = defaultdict(lambda: defaultdict(list))

        for seed in range(a.seeds):
            split = protocols.make_split(data, ds, seed=seed)
            m = protocols.masks(data, split)
            for s in SUBSETS:
                if s not in m:
                    continue
                sel = m[s]
                per_seed[s]["subjects"].append(int(len(np.unique(subj[sel]))))
                per_seed[s]["records"].append(int(len(np.unique(rec[sel]))))
                per_seed[s]["beats"].append(int(sel.sum()))

        out[ds] = {
            s: {k: dict(mean=float(np.mean(v)), min=int(np.min(v)),
                        max=int(np.max(v)), seed0=int(v[0]))
                for k, v in d.items()}
            for s, d in per_seed.items()
        }
        out[ds]["totals"] = dict(
            subjects=int(len(np.unique(subj))),
            records=int(len(np.unique(rec))),
            beats=int(len(subj)))
        print(f"=== {ds} ===", flush=True)
        for s in SUBSETS:
            if s in out[ds]:
                r = out[ds][s]
                print(f"  {s:10s} subj={r['subjects']['seed0']:4d} "
                      f"rec={r['records']['seed0']:4d} "
                      f"beats={r['beats']['seed0']:6d}"
                      f"   (records across seeds "
                      f"{r['records']['min']}-{r['records']['max']})", flush=True)

    # ---- Heartprint session structure (R1.10)
    meta_p = os.path.join(a.proc, "heartprint_meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        sess = defaultdict(set)
        for r, info in (meta.get("records") or {}).items():
            s = info.get("session") if isinstance(info, dict) else None
            if s is not None:
                sess[str(s)].add(r)
        out["heartprint_sessions"] = {
            "records_per_session": {k: len(v) for k, v in sorted(sess.items())},
            "date_field_available": any(
                isinstance(i, dict) and i.get("date")
                for i in (meta.get("records") or {}).values()),
            "note": ("Heartprint distributes session labels (1, 2, 3L, 3R) but "
                     "the public release does not carry per-record acquisition "
                     "dates, so the elapsed-time distribution cannot be computed "
                     "from the data and is reported qualitatively."),
        }
        print("\n=== heartprint sessions ===")
        print(json.dumps(out["heartprint_sessions"], indent=1))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
