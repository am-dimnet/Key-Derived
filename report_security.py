import glob
import json

import numpy as np

from paths import ROOT as _ROOT

rows = {}
for p in glob.glob(f"{_ROOT}/results/security_*proposed*.json"):
    r = json.load(open(p))
    rows.setdefault(r["dataset"], []).append(r)


def m(rs, *path):
    v = []
    for r in rs:
        c = r
        for k in path:
            c = c.get(k) if isinstance(c, dict) else None
            if c is None:
                break
        if isinstance(c, (int, float)):
            v.append(c)
    return float(np.mean(v)) if v else float("nan")


hdr = ("dataset", "linkAUC", "Dsys", "replay%", "bitsamp%", "zeroFAR%",
       "FD_ASR%", "stale%", "fresh%", "xfer%")
print(f"{hdr[0]:12s}" + "".join(f"{h:>9s}" for h in hdr[1:]))
for d, rs in sorted(rows.items()):
    vals = [
        m(rs, "unlinkability", "link_auc"),
        m(rs, "unlinkability", "link_dsys"),
        m(rs, "attack_template_only", "replay_stolen_template_pct"),
        m(rs, "attack_template_only", "bit_sampling_pct"),
        m(rs, "attack_stolen_key", "zero_effort_impostor_far_pct"),
        m(rs, "attack_full_disclosure", "full_disclosure_asr_pct"),
        m(rs, "revocation", "stale_template_acceptance_pct"),
        m(rs, "revocation", "fresh_template_acceptance_pct"),
        m(rs, "revocation", "cross_version_transfer_pct"),
    ]
    print(f"{d:12s}" + "".join(f"{v:9.2f}" for v in vals))

print("\n--- linkability vs number of observed revoked templates (proposed) ---")
for d, rs in sorted(rows.items()):
    curve = rs[0]["revocation"]["linkability_vs_k"]
    print(f"  {d:12s}" + "  ".join(
        f"k={c['k']}:{c['link_auc']:.3f}" for c in curve))

print("\n--- attribute leakage ---")
for d, rs in sorted(rows.items()):
    al = rs[0].get("attribute_leakage") or rs[0].get("attribute_leakage_error")
    print(f"  {d}: {json.dumps(al)[:300]}")

print("\n--- full-disclosure attack budget ---")
d0 = sorted(rows)[0]
print(" ", json.dumps(rows[d0][0]["attack_full_disclosure"].get("budget")))
print("  mean iterations to succeed:",
      round(m(rows[d0], "attack_full_disclosure", "mean_iterations"), 1))
print("\nthresholds fixed on:", rows[d0][0].get("threshold_source"))
