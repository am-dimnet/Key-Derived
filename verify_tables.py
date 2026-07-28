"""Recompute every headline number in the paper from the released result files.

This is the quickest way to check the paper against its data: it needs no GPU,
no datasets and no trained weights, only the JSON in results/.  Each check
prints the value in the manuscript beside the value recomputed here.

    python verify_tables.py

Exit status is 0 when every check agrees to the printed precision, 1 otherwise.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "results", "tables")
# evaluate.py and security.py write these flat into results/; the shipped tree
# groups them into per_run/ and scores/ for readability.  Accept either.
_R = os.path.join(HERE, "results")
RUN = os.path.join(_R, "per_run") if os.path.isdir(os.path.join(_R, "per_run")) else _R
SCORES = os.path.join(_R, "scores") if os.path.isdir(os.path.join(_R, "scores")) else _R

DS = ("ecgid", "heartprint", "ptbdb")
LABEL = {"ecgid": "ECG-ID", "heartprint": "Heartprint", "ptbdb": "PTB"}

fails = []


def check(name, paper, got, tol=None):
    """Compare a value printed in the manuscript with the recomputed one.

    The default tolerance is half a unit in the last place the manuscript
    prints, so a value quoted to one decimal is checked to +/-0.05 and one
    quoted to three decimals to +/-0.0005.
    """
    if tol is None:
        s = f"{paper}"
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
    ok = abs(paper - got) <= tol
    if not ok:
        fails.append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:52s} paper={paper:9.3f}  recomputed={got:9.3f}")


def load(p):
    with open(p) as f:
        return json.load(f)


def per_run(pattern):
    return [load(p) for p in sorted(glob.glob(os.path.join(RUN, pattern)))]


print("=" * 78)
print("Table V — same-session verification (EER %, mean over 5 seeds)")
print("=" * 78)
main = {r["method"]: r for r in load(os.path.join(TAB, "table_main.json"))["table"]}
PAPER_V = {
    ("proposed", "ecgid"): 8.37, ("proposed", "heartprint"): 10.56, ("proposed", "ptbdb"): 10.54,
    ("cnn", "ecgid"): 6.82, ("cnn", "heartprint"): 10.00, ("cnn", "ptbdb"): 8.80,
    ("cnn_biohash", "ecgid"): 7.79, ("cnn_biohash", "heartprint"): 10.27, ("cnn_biohash", "ptbdb"): 8.53,
    ("cnn_randproj", "ecgid"): 7.17, ("cnn_randproj", "heartprint"): 10.87, ("cnn_randproj", "ptbdb"): 9.95,
    ("cnn_compsens", "ecgid"): 7.69, ("cnn_compsens", "heartprint"): 10.62, ("cnn_compsens", "ptbdb"): 8.51,
}
for (m, d), paper in PAPER_V.items():
    check(f"{main[m]['label']} / {LABEL[d]}", paper, main[m][d]["eer"])

# the ECG-ID control row is the re-run at the matched template length L=512,
# not the superseded L=256 entry that table_main.json still carries
ctrl = per_run("result_ecgid_cnn_posthoc_ours_L512_s*.json")
check("Ours, post-hoc (control) / ECG-ID  [L=512]", 6.75, float(np.mean([r["eer"] for r in ctrl])))
check("Ours, post-hoc (control) / Heartprint", 11.42, main["cnn_posthoc_ours"]["heartprint"]["eer"])
check("Ours, post-hoc (control) / PTB", 9.74, main["cnn_posthoc_ours"]["ptbdb"]["eer"])

print()
print("=" * 78)
print("Table VII — revocability, RevScore = mean_seeds[1 - |new-old|/old]")
print("=" * 78)
rev = load(os.path.join(TAB, "revocability.json"))
base = load(os.path.join(TAB, "baseline_security.json"))
ctl = load(os.path.join(TAB, "control_security.json"))
for d, paper in zip(DS, (0.556, 0.939, 0.924)):
    check(f"Proposed / {LABEL[d]}", paper, rev[f"{d}/proposed"]["revscore"])
for d, paper in zip(DS, (0.919, 0.977, 0.951)):
    check(f"CNN+BioHashing / {LABEL[d]}", paper, base[f"{d}/cnn_biohash"]["revscore"])
for d, paper in zip(DS, (0.866, 0.916, 0.937)):
    check(f"Ours, post-hoc (control) / {LABEL[d]}", paper, ctl[f"{d}/cnn_posthoc_ours"]["revscore"])

print()
print("=" * 78)
print("Table VIII — identity recovery and unlinkability")
print("=" * 78)
irr = load(os.path.join(TAB, "irr.json"))
irr_c = load(os.path.join(TAB, "irr_control.json"))
fpl = load(os.path.join(TAB, "fixedproj_link.json"))
for d, paper in zip(DS, (77.8, 60.5, 68.9)):
    check(f"IRR, proposed / {LABEL[d]}", paper, irr[f"{d}/proposed"]["irr_mean"])
for d, paper in zip(DS, (77.7, 64.1, 75.2)):
    check(f"IRR, unprotected CNN / {LABEL[d]}", paper, irr[f"{d}/cnn"]["irr_mean"])
for d, paper in zip(DS, (78.8, 61.1, 71.2)):
    check(f"IRR, post-hoc control / {LABEL[d]}", paper, irr_c[f"{d}/cnn_posthoc_ours"]["irr_mean"])
for d, paper in zip(DS, (0.487, 0.540, 0.493)):
    check(f"Linkability AUC, proposed / {LABEL[d]}", paper, fpl[f"{d}/proposed"]["link_auc"])
for d in DS:
    check(f"Linkability AUC, w/o key projection / {LABEL[d]}", 1.000,
          fpl[f"{d}/fixed_proj"]["link_auc"])

print()
print("=" * 78)
print("Table IX — ablation, EER % on all three datasets")
print("=" * 78)
for m, papers in (("no_metric", (6.81, 11.53, 10.67)),
                  ("no_temp", (9.16, 10.95, 10.13)),
                  ("fixed_proj", (7.99, 10.47, 10.85))):
    for d, paper in zip(DS, papers):
        check(f"{m} / {LABEL[d]}", paper, main[m][d]["eer"])

print()
print("=" * 78)
print("Table X — template length sweep on ECG-ID")
print("=" * 78)
tl = load(os.path.join(TAB, "template_length.json"))
for L, eer, link, ms in ((64, 7.99, 0.498, 0.146), (128, 7.08, 0.485, 0.147),
                         (256, 7.91, 0.543, 0.146), (512, 8.28, 0.507, 0.148)):
    check(f"L={L} EER", eer, tl[str(L)]["eer"]["mean"])
    check(f"L={L} linkability AUC", link, tl[str(L)]["link_auc"]["mean"])
    check(f"L={L} per-beat time (ms)", ms, tl[str(L)]["ms"]["mean"])

print()
print("=" * 78)
print("Revocation security (Section IV-E prose)")
print("=" * 78)
for d, fresh in zip(DS, (95.2, 71.2, 80.8)):
    R = [load(p)["revocation"] for p in sorted(glob.glob(os.path.join(RUN, f"security_{d}_proposed_*.json")))]
    check(f"stale-template acceptance / {LABEL[d]}", 0.0,
          float(np.mean([x["stale_template_acceptance_pct"] for x in R])))
    check(f"fresh re-enrolled acceptance / {LABEL[d]}", fresh,
          float(np.mean([x["fresh_template_acceptance_pct"] for x in R])))
    check(f"cross-version transfer / {LABEL[d]}", 0.0,
          float(np.mean([x["cross_version_transfer_pct"] for x in R])))

k = load(os.path.join(TAB, "irr_vs_k.json"))
for d, a, b in zip(DS, (1.5, 2.1, 2.0), (1.7, 1.8, 2.3)):
    check(f"IRR given k=1 revoked / {LABEL[d]}", a, k[f"{d}/k1"]["irr_mean"])
    check(f"IRR given k=5 revoked / {LABEL[d]}", b, k[f"{d}/k5"]["irr_mean"])

print()
print("=" * 78)
print("Fig. 2 — pooled DET operating points (needs results/scores/)")
print("=" * 78)
if glob.glob(os.path.join(SCORES, "*.npz")):
    def tpr_at(ds, meth, far_target):
        fs = sorted(glob.glob(os.path.join(SCORES, f"scores_{ds}_{meth}_L*_s*.npz")))
        s = np.concatenate([np.load(f)["score"].astype(float) for f in fs])
        l = np.concatenate([np.load(f)["label"].astype(int) for f in fs])
        o = np.argsort(-s); l = l[o]
        far = np.cumsum(1 - l) / (len(l) - l.sum())
        tpr = np.cumsum(l) / l.sum()
        return float(tpr[far <= far_target].max() * 100)
    check("PTB proposed TPR @ FAR=1%", 58.4, tpr_at("ptbdb", "proposed", 0.01))
    check("PTB BioHashing TPR @ FAR=1%", 71.1, tpr_at("ptbdb", "cnn_biohash", 0.01))
    check("PTB proposed TPR @ FAR=0.1%", 31.8, tpr_at("ptbdb", "proposed", 0.001))
    check("PTB BioHashing TPR @ FAR=0.1%", 47.6, tpr_at("ptbdb", "cnn_biohash", 0.001))
    check("Heartprint proposed TPR @ FAR=1%", 55.3, tpr_at("heartprint", "proposed", 0.01))
    check("Heartprint BioHashing TPR @ FAR=1%", 59.8, tpr_at("heartprint", "cnn_biohash", 0.01))
else:
    print("  no score arrays found — skipping the DET checks")

print()
print("=" * 78)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("All checks agree with the manuscript.")
sys.exit(0)
