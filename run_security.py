"""Run the template-protection and attack experiments over the trained grid.

Covers the reviewer requests that go beyond verification accuracy:
  R1.6   attack methodology, described operationally per disclosure setting
  R1.7   revocation security (stale-template invalidation, multi-template
         linkability, cross-version transfer) on top of RevScore
  R1.12  unlinkability reported with both the AUC and the system-level D_sys
  R2.1   health-attribute leakage from embedding vs protected template
"""
import sys
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from paths import ROOT as _ROOT

PY = sys.executable
CODE = f"{_ROOT}/code"
ROOT = f"{_ROOT}"


def job(dataset, variant, seed, steps, skip_attacks):
    tag = f"{dataset}_{variant}_s{seed}"
    log = os.path.join(ROOT, "logs", f"security_{tag}.log")
    if not os.path.exists(os.path.join(ROOT, "runs",
                                       f"{dataset}_{variant}_L256_s{seed}",
                                       "best.pt")):
        # the tuned config may use a different template length
        import glob
        hits = glob.glob(os.path.join(ROOT, "runs",
                                      f"{dataset}_{variant}_L*_s{seed}", "best.pt"))
        if not hits:
            return f"  skip  {tag} (no checkpoint)"
        L = os.path.basename(os.path.dirname(hits[0])).split("_L")[1].split("_")[0]
    else:
        L = "256"
    args = [PY, os.path.join(CODE, "security.py"), "--dataset", dataset,
            "--variant", variant, "--seed", str(seed), "--L", L,
            "--attack_steps", str(steps)]
    if skip_attacks:
        args.append("--skip_attacks")
    env = dict(os.environ, OMP_NUM_THREADS="4", TORCH_THREADS="4")
    with open(log, "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        return f"  FAIL  {tag} (see {log})"
    return f"  ok    {tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecgid,heartprint,ptbdb")
    ap.add_argument("--variants", default="proposed,no_metric,no_temp,fixed_proj")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--par", type=int, default=12)
    ap.add_argument("--attack_steps", type=int, default=300)
    ap.add_argument("--skip_attacks", action="store_true")
    a = ap.parse_args()

    jobs = [(d, v, s) for d in a.datasets.split(",")
            for v in a.variants.split(",") for s in range(a.seeds)]
    print(f"=== {len(jobs)} security runs, {a.par} in parallel ===", flush=True)
    with ThreadPoolExecutor(max_workers=a.par) as ex:
        for msg in ex.map(
                lambda t: job(*t, a.attack_steps, a.skip_attacks), jobs):
            print(msg, flush=True)

    # ---- collect
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "security_*.json"))):
        with open(p) as f:
            r = json.load(f)
        out.setdefault(f"{r['dataset']}/{r['variant']}", []).append(r)

    summary = {}
    for k, rs in out.items():
        def mean(path, default=float("nan")):
            vals = []
            for r in rs:
                cur = r
                for part in path.split("."):
                    cur = cur.get(part) if isinstance(cur, dict) else None
                    if cur is None:
                        break
                if isinstance(cur, (int, float)):
                    vals.append(cur)
            return sum(vals) / len(vals) if vals else default
        summary[k] = dict(
            n_seeds=len(rs),
            link_auc=mean("unlinkability.link_auc"),
            link_dsys=mean("unlinkability.link_dsys"),
            stale_acceptance_pct=mean("revocation.stale_template_acceptance_pct"),
            fresh_acceptance_pct=mean("revocation.fresh_template_acceptance_pct"),
            impostor_acceptance_pct=mean("revocation.impostor_acceptance_pct"),
            cross_version_transfer_pct=mean("revocation.cross_version_transfer_pct"),
            replay_stolen_template_pct=mean("attack_template_only.replay_stolen_template_pct"),
            bit_sampling_pct=mean("attack_template_only.bit_sampling_pct"),
            zero_effort_impostor_far_pct=mean("attack_stolen_key.zero_effort_impostor_far_pct"),
            full_disclosure_asr_pct=mean("attack_full_disclosure.full_disclosure_asr_pct"),
        )
    path = os.path.join(ROOT, "results", "tables", "security_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== security summary ===", flush=True)
    for k, v in sorted(summary.items()):
        print(f"{k}: link_auc={v['link_auc']:.3f} D_sys={v['link_dsys']:.3f} "
              f"stale_acc={v['stale_acceptance_pct']:.1f}% "
              f"FD_asr={v['full_disclosure_asr_pct']:.1f}%", flush=True)
    print("\nwrote", path, flush=True)


if __name__ == "__main__":
    main()
