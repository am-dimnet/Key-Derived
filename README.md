# Key-Derived Cancelable Template Learning for Privacy-Preserving ECG Authentication

Code and released results for the manuscript submitted to *IEEE Transactions on
Biometrics, Behavior, and Identity Science* (TBIOM-2026-05-0140).

Released at <https://github.com/am-dimnet/Key-Derived/releases/tag/ECG>.

Everything needed to recompute the numbers in the paper is in this directory.
Retraining from scratch additionally needs the three public databases and a GPU.

---

## Quickest check: recompute the paper from the released results

No GPU, no datasets, no trained weights.

```bash
pip install numpy
python verify_tables.py
```

This recomputes 84 values — every headline number in Tables V, VII, VIII, IX and
X, the revocation-security figures quoted in Section IV-E, and the pooled DET
operating points behind Fig. 2 — and prints each beside the value printed in the
manuscript. Tolerance is half a unit in the last decimal the paper shows. Exit
status is 0 only if every check agrees.

## Regenerating the figures

```bash
pip install numpy matplotlib scipy
cd figures && python make_det.py && python make_figs.py
```

`figures/output/FigDET.pdf` is Fig. 2 of the manuscript. It renders pixel-identical
to the file used in the paper; the PDF bytes differ only in the embedded creation
timestamp.

`make_figs.py` also emits a trade-off scatter (`Fig10.pdf`) and a bar chart
(`Fig3.png`). Neither appears in the current manuscript — the trade-off figure was
removed because Tables V and VIII carry its content — but both are kept here
because they were used while preparing the revision.

---

## What is in this directory

```
verify_tables.py          recompute every headline number from results/
code/                     the pipeline: preprocessing, model, training, evaluation, attacks
figures/                  figure-generation scripts; output lands in figures/output/
data_acquisition/         scripts that fetch the three databases
results/
  tables/                 aggregated JSON, one file per table in the paper
  per_run/                230 per-run files: 170 result_*.json and 60 security_*.json
  scores/                 170 score arrays (genuine/impostor scores per run) for the DET curves
  dataset_meta/           preprocessing ledger per database: what was read, retained, discarded
  best_configs.json       the configuration the equal-budget search selected per system
```

Every file under `results/` was written by the scripts in `code/`; none of the
values was edited by hand. The only manual step was filing: `evaluate.py` and
`security.py` write their per-run JSON and score arrays flat into `results/`, and
they were moved into `per_run/` and `scores/` here so the directory is readable.
`verify_tables.py` and `figures/make_det.py` accept either layout, so a
from-scratch run needs no reorganisation.

---

## Reproducing from scratch

### 1. Configure a working root

Every path derives from one root. Set it if you want the data and checkpoints
somewhere other than this directory:

```bash
export ARTICLE4_ROOT=/path/with/several/GB/free
```

Unset, the root is this directory. `code/paths.py` is the only place this is
resolved.

### 2. Obtain the databases

ECG-ID and the PTB Diagnostic ECG Database come from PhysioNet; Heartprint comes
from its authors. `data_acquisition/` holds the fetch scripts used here. Place
them under `$ARTICLE4_ROOT/data/raw/{ecgid,heartprint,ptbdb}`.

### 3. Preprocess

```bash
python code/prep.py --dataset ecgid      --raw $ARTICLE4_ROOT/data/raw/ecgid
python code/prep.py --dataset heartprint --raw $ARTICLE4_ROOT/data/raw/heartprint
python code/prep.py --dataset ptbdb      --raw $ARTICLE4_ROOT/data/raw/ptbdb
```

Writes `$ARTICLE4_ROOT/data/proc/<dataset>.npz` plus a metadata file recording
the retention ledger. The pipeline is deterministic: two independent rebuilds on
different machines produced identical counts (89/307/7392 subjects/records/beats
on ECG-ID, 199/1197/21765 on Heartprint, 110/361/49064 on PTB). Compare
`data/proc/*_meta.json` against `results/dataset_meta/` to confirm.

### 4. Train and evaluate

```bash
python code/run_final.py --datasets ecgid,heartprint,ptbdb --seeds 5 --epochs 80 --par 8 --l4 5.0
```

Trains every variant at its tuned configuration over five seeds and evaluates all
methods. On two RTX 6000D this took roughly a day. `code/sweep.py` reruns the
equal-budget hyper-parameter search that produced `results/best_configs.json`.

### 5. Security, revocation and the remaining tables

```bash
python code/run_security.py --datasets ecgid,heartprint,ptbdb --seeds 5 --par 8 --attack_steps 300
python code/fill_tables.py            # baseline security columns
python code/control_security.py       # the single-variable control
python code/irr.py                    # identity recovery
python code/irr_vs_k.py               # identity recovery vs number of revoked templates
python code/crosssession_all.py       # cross-session on all three databases
python code/length_sweep.py           # template-length sweep
python code/protocol_counts.py        # per-subset subject/record/beat counts
python code/extra_tables.py           # revocability table
python code/fix_fixedproj_link.py     # corrected fixed-projection linkability
python code/irr.py --methods cnn_posthoc_ours --out $ARTICLE4_ROOT/results/tables/irr_control.json
python code/aggregate.py              # collect everything into results/tables/
```

---

## Which script produces which table

| Paper | Content | Script | Result file |
|---|---|---|---|
| Table II | Implementation details | — | `results/best_configs.json` |
| Table III | Threat settings | — | (specification, not measured) |
| Table IV | Dataset and protocol counts | `prep.py`, `protocol_counts.py` | `results/dataset_meta/`, `tables/protocol_counts.json` |
| Table V | Same-session verification | `run_final.py` → `evaluate.py` | `tables/table_main.json`, `per_run/result_*.json` |
| Table VI | Cross-session | `crosssession_all.py` | `tables/cross_session_all.json` |
| Table VII | Revocability | `extra_tables.py`, `fill_tables.py`, `control_security.py` | `tables/revocability.json`, `baseline_security.json`, `control_security.json` |
| Table VIII | Unlinkability, IRR, attacks | `run_security.py`, `fill_tables.py`, `irr.py`, `fix_fixedproj_link.py` | `tables/security_summary.json`, `baseline_security.json`, `irr.json`, `irr_control.json`, `fixedproj_link.json` |
| Table IX | Ablation, three datasets | `run_final.py`, `run_security.py` | `tables/table_main.json`, `security_summary.json` |
| Table X | Template-length sweep | `length_sweep.py` | `tables/template_length.json` |
| Fig. 1 | Framework diagram | — | (drawn, not generated) |
| Fig. 2 | DET curves | `figures/make_det.py` | `results/scores/*.npz` |
| Sec. IV-E | Revocation security | `run_security.py`, `irr_vs_k.py` | `per_run/security_*.json`, `tables/irr_vs_k.json` |

---

## Known issues

`KNOWN_ISSUES.md` records the defects an independent audit found in this code and
what was done about each. Four of the five that bore on reported numbers have been
fixed and every affected measurement re-run; the fifth was a scope mismatch between
the paper and the code and was resolved by correcting the paper. Read it before
relying on the security tables, and note that the results shipped here come from
the corrected pipeline.

## Two corrections made during the revision

Both are in the code as shipped; they are recorded here because they changed
numbers that appeared in the submitted version.

**Template length in the post-hoc control.** `evaluate.py` used to resolve the
template length from the encoder checkpoint directory. The shared CNN backbone is
stored at `L=256` while the tuned proposed model on ECG-ID uses `L=512`, so the
control — whose entire purpose is a matched comparison — had been measured at half
the template length of the system it controls for. For post-hoc methods the length
is a free parameter of the transform, not a property of the backbone, so the
requested value now survives. The ECG-ID control moved from 7.26% to 6.75% EER,
widening the gap against the proposed method.

**RevScore definition.** `control_security.py` initially computed an EER ratio
clamped at 1.0 rather than Eq. (2) of the paper, `1 - |new-old|/old` averaged over
seeds, which is what every other row of the revocability table uses. Corrected,
the control's RevScore is 0.866 / 0.915 / 0.937 rather than 0.896 / 0.981 / 0.989.
The corrected values change the conclusion: the control re-issues more stably than
the jointly optimised model on ECG-ID and PTB but less stably on Heartprint, so
post-hoc protection does not dominate on that axis.
`results/tables/control_security.json` retains the superseded value under
`revscore_OLD_ratio_formula` for traceability.

---

## Notes on the results as reported

- The five seeds redraw the subject partition as well as varying initialisation
  and mini-batch composition, so a reported standard deviation mixes partition and
  optimisation variability. The subject-level bootstrap interval (500 repetitions,
  fixed partition) is reported alongside because it answers the other question.
- Subject pools for training, validation and evaluation are disjoint and record
  pools within a subject are disjoint. `protocols.py` asserts on the materialised
  arrays after segmentation that no recording feeds two subsets; the assertion
  passes for every dataset and seed.
- The transform-known and full-disclosure attacks were not run for the post-hoc
  control, which is why those cells are dashes in Table VIII.

## Environment

Measurements in the paper were produced with Python 3.10, PyTorch 2.8.0 and CUDA
12.8 on an NVIDIA RTX 6000D (85 GiB). `verify_tables.py` and the figure scripts
need only NumPy, Matplotlib and SciPy and run on CPU.

## Licence

MIT, see `LICENSE`. The three databases keep their own licences and are not
redistributed here.
