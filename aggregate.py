"""Collect per-run JSON results into the manuscript tables."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from metrics import paired_wilcoxon

from paths import ROOT as _ROOT

METHOD_ORDER = ["feature_svm", "cnn", "siamese", "cnn_biohash",
                "cnn_randproj", "cnn_compsens", "cnn_posthoc_ours", "proposed",
                "no_metric", "no_temp", "fixed_proj"]
METHOD_LABEL = {
    "feature_svm": "Feature+SVM",
    "cnn": "CNN (unprotected embedding)",
    "siamese": "Siamese (unprotected embedding)",
    "cnn_biohash": "CNN + BioHashing (post-hoc)",
    "cnn_randproj": "CNN + Random Projection (post-hoc)",
    "cnn_compsens": "CNN + Compressive Sensing (post-hoc)",
    "cnn_posthoc_ours": "Ours, post-hoc transform (control)",
    "proposed": "Proposed (jointly optimised)",
    "no_metric": "w/o metric loss",
    "no_temp": "w/o template regularisation",
    "fixed_proj": "w/o key-derived projection",
}
DATASET_LABEL = {"ecgid": "ECG-ID", "heartprint": "Heartprint", "ptbdb": "PTB"}


def load_results(res_dir):
    rows = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(res_dir, "result_*.json"))):
        with open(p) as f:
            r = json.load(f)
        rows[(r["dataset"], r["method"])].append(r)
    return rows


def mean_sd(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def table_main(rows, datasets, methods):
    """Table I: EER / AUC / TPR@FAR=1% per dataset and method."""
    out = []
    for meth in methods:
        rec = {"method": meth, "label": METHOD_LABEL.get(meth, meth)}
        for ds in datasets:
            rs = rows.get((ds, meth), [])
            if not rs:
                rec[ds] = None
                continue
            eer_m, eer_s = mean_sd([r["eer"] for r in rs])
            auc_m, auc_s = mean_sd([r["auc"] for r in rs])
            tpr_m, tpr_s = mean_sd([r["tpr_at_far1"] for r in rs])
            cis = [r.get("eer_ci95", [np.nan, np.nan]) for r in rs]
            rec[ds] = dict(
                n_seeds=len(rs),
                L=sorted({r.get("L", 256) for r in rs}),
                eer=eer_m, eer_sd=eer_s,
                auc=auc_m, auc_sd=auc_s,
                tpr_at_far1=tpr_m, tpr_sd=tpr_s,
                eer_ci95_mean=[float(np.nanmean([c[0] for c in cis])),
                               float(np.nanmean([c[1] for c in cis]))],
            )
        out.append(rec)
    return out


def significance(rows, datasets, ref="proposed", others=None):
    """Paired Wilcoxon on subject-level EER; seeds averaged per subject first."""
    others = others or ["cnn", "siamese", "cnn_biohash", "cnn_randproj",
                        "cnn_compsens", "cnn_posthoc_ours", "feature_svm"]
    out = {}
    for ds in datasets:
        base = rows.get((ds, ref), [])
        if not base:
            continue
        a = defaultdict(list)
        for r in base:
            for k, v in r.get("per_subject_eer", {}).items():
                a[int(k)].append(v)
        A = {k: float(np.mean(v)) for k, v in a.items()}
        for o in others:
            rs = rows.get((ds, o), [])
            if not rs:
                continue
            b = defaultdict(list)
            for r in rs:
                for k, v in r.get("per_subject_eer", {}).items():
                    b[int(k)].append(v)
            B = {k: float(np.mean(v)) for k, v in b.items()}
            out[f"{ds}:{ref}_vs_{o}"] = paired_wilcoxon(A, B)
    out["_pairing"] = ("paired observations are per-subject EER values on the "
                       "identical subject set; the seeded runs are averaged per "
                       "subject before pairing, so the test compares systems "
                       "rather than optimisation noise. No multiplicity "
                       "correction is applied.")
    return out


def fmt_markdown(tbl, datasets):
    hdr = "| Method | " + " | ".join(
        f"{DATASET_LABEL.get(d, d)} EER(%) | {DATASET_LABEL.get(d, d)} AUC | "
        f"{DATASET_LABEL.get(d, d)} TPR@FAR1%" for d in datasets) + " |"
    sep = "|" + "---|" * (1 + 3 * len(datasets))
    lines = [hdr, sep]
    for rec in tbl:
        cells = [rec["label"]]
        for d in datasets:
            v = rec.get(d)
            if not v:
                cells += ["--", "--", "--"]
            else:
                cells += [f"{v['eer']:.2f} ± {v['eer_sd']:.2f}",
                          f"{v['auc']:.4f}", f"{v['tpr_at_far1']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{_ROOT}/results")
    ap.add_argument("--out", default=f"{_ROOT}/results/tables")
    ap.add_argument("--L", type=int, default=256)
    a = ap.parse_args()

    rows = load_results(a.results)
    datasets = [d for d in ("ecgid", "heartprint", "ptbdb")
                if any(k[0] == d for k in rows)]
    methods = [m for m in METHOD_ORDER if any(k[1] == m for k in rows)]

    os.makedirs(a.out, exist_ok=True)
    main_tbl = table_main(rows, datasets, methods)
    sig = significance(rows, datasets)

    with open(os.path.join(a.out, "table_main.json"), "w") as f:
        json.dump(dict(table=main_tbl, datasets=datasets,
                       significance=sig), f, indent=2)
    md = fmt_markdown(main_tbl, datasets)
    with open(os.path.join(a.out, "table_main.md"), "w") as f:
        f.write("# Authentication performance\n\n" + md + "\n\n")
        f.write("## Paired Wilcoxon (subject-level EER)\n\n")
        for k, v in sig.items():
            if k.startswith("_"):
                continue
            f.write(f"- `{k}`: n={v['n_pairs']}, W={v['statistic']:.1f}, "
                    f"p={v['p_value']:.3g}, median diff={v.get('median_diff', float('nan')):.3f}\n")
        f.write("\n" + sig["_pairing"] + "\n")
    print(md)
    print("\nwrote", os.path.join(a.out, "table_main.md"))


if __name__ == "__main__":
    main()
