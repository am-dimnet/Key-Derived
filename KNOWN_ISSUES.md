# Known issues

An independent audit of this package against the manuscript found the defects
below. Four of the five that bore on reported numbers have since been fixed and
every affected measurement re-run; the results shipped here come from the
corrected pipeline. Each entry records what was wrong, what was done, and what it
changed. Nothing has been removed from this file: an issue that was fixed is
marked FIXED and keeps its original description, so a reader can see what the
earlier numbers were subject to.

## Status at a glance

| # | Defect | Status | Effect on the paper |
|---|---|---|---|
| 1 | Key scoped per application-version, not per user | RESOLVED IN THE PAPER | The manuscript now claims application-version scoping and states the consequences; no per-user claim remains |
| 2 | Threshold and attack used different projections | FIXED, RE-RUN | Stored-template ASR unchanged; linkability and IRR moved by a few points |
| 3 | Undisclosed distillation term | DISCLOSED | Now in Eq. (12) and Table II; retraining with the weight at zero moves EER by at most 1.05 points |
| 4 | Distillation teacher shared across seeds | FIXED, RE-RUN | Proposed EER 8.03/11.42/10.35 to 8.37/10.56/10.54; stored-template ASR 59.4/16.2/16.4 to 71.9/5.8/35.9 |
| 5 | Implemented metric loss differed from Eq. (13) | RESOLVED IN THE PAPER | Section III-E now gives all three implemented losses and Table II records which was selected per database |

Items marked FIXED were corrected in the code in this package and every dependent
measurement was recomputed. Items marked RESOLVED IN THE PAPER were mismatches
between the manuscript and the code in which the code was right.

---

## 1. The key is application-scoped, not per-user  — RESOLVED IN THE PAPER

`derive_seed(app, user, version, bind_user, bind_version)` in `code/models.py`
supports the user-application-version triple the paper describes. But
`CancelableLayer`, `ProtectedNet` and all four post-hoc transforms default to
`bind_user=False`, and no caller anywhere sets it to `True` or supplies a subject
identifier — `evaluate.score_matrix` passes `user=""`.

Every reported number therefore uses one application-level key shared by all
enrolled users, not a key derived per user. Sections III-C and III-F, Table I and
Section II-D all state the per-user property.

This is not cosmetic. `security.attack_template_only` builds its candidate
distribution as `p = Pe.mean(0)`, pooled over all identities, which only carries
information about a target because every identity shares one projection. Under a
per-user key that attack would be uninformative, so the stored-template exposure
figures in Table VIII (59.4 / 16.2 / 16.4 %) are a property of the key scope
actually used.

Resolution: either thread the subject id through the layer and set
`bind_user=True`, then re-run training, evaluation and all of Sections IV-D to
IV-F; or restate the manuscript to say the experiments simulate an
application-scoped key and re-word the revocation-scoping claims.

## 2. Threshold and attack use different projections  — FIXED, ALL AFFECTED RESULTS RE-RUN

`CancelableLayer.rekey` derives its seed with `bind_user=False`. Every security,
IRR and sweep script instead calls `derive_seed("app-A", version=0)`, which takes
the default `bind_user=True` and produces a *different* seed, hence a different
`R` and `b`.

In `security.main`, `tau` comes from `validation_threshold`, which runs the
model's own trained layer, while `Pe`, `Tq`, the unlinkability pair, the
revocation versions and the IRR features are built from the re-derived
projection. `fill_tables.py` has the same asymmetry for the proposed and ablation
rows, while the post-hoc baseline rows reuse one transform object throughout and
are consistent.

So the rows the paper reports as more exposed than the post-hoc baselines are
exactly the rows whose operating threshold does not belong to the projection
being attacked. Table VIII, Table X and the Section IV-E acceptance percentages
depend on this.

Resolution: have one code path own the transform — expose `R`, `b` from the
loaded `CancelableLayer` and use them everywhere, or pass `bind_user=False`
explicitly at every `derive_seed` call site — then re-run `run_security.py`,
`fill_tables.py`, `irr.py`, `irr_vs_k.py` and `length_sweep.py`.

## 3. Undisclosed fourth loss term  — DISCLOSED, AND MEASURED WITHOUT IT

`train.py` defines `distill_loss`, a similarity-distillation term against an
unprotected CNN teacher. `run_final.py` defaults `--l4 5.0` and attaches the
teacher to `proposed`, `no_metric`, `no_temp` and `fixed_proj`, and the README's
reproduce command carries `--l4 5.0`, so every reported protected model was
trained with it.

Equation (12) of the paper has three terms and neither Section III-E nor Table II
mentions distillation. The protected encoder is therefore partly a distillation of
an unprotected recognition model, which bears on how protected-space learning is
framed.

Resolution: add the term to Eq. (12) and Table II with its weight, or retrain the
final grid with `--l4 0` and re-report Tables V, VII, VIII and IX.

## 4. The distillation teacher leaks evaluation subjects across seeds  — FIXED, ALL PROTECTED VARIANTS RETRAINED

`run_final.train_teacher` trains one teacher per dataset at seed 0 and reuses it
for all five seeds. `protocols.make_split` redraws the subject partition per seed,
so the seed-0 teacher was trained on subjects that are enrolment or test subjects
under seeds 1 to 4 — on ECG-ID roughly half the evaluation subjects in
expectation — and its pairwise geometry is distilled into the protected encoder
for those seeds.

`protocols.assert_no_leakage` cannot see this: it checks record and subject pools
within a single split. Section IV-A states that enrolled identities never appear
in training. The CNN, Siamese and post-hoc baselines are trained per seed and are
unaffected, so the contamination is asymmetric in exactly the comparison Table V
makes.

Resolution: train one teacher per (dataset, seed) on that seed's own split, or
drop distillation; then re-run seeds 1 to 4 for the protected variants.

## 5. The implemented metric loss is not Equation (13)  — RESOLVED IN THE PAPER

Section III-E specifies a supervised contrastive loss with cosine similarity and a
temperature. `train.py` implements three alternatives — Euclidean-margin
contrastive, batch-hard triplet, and a margin on the normalised inner product —
none of which has the log-softmax-over-positives form of Eq. (13), and no
temperature parameter exists in the codebase.

`results/best_configs.json` further shows the equal-budget search selected a
different one per dataset: contrastive on ECG-ID, hamming on Heartprint, triplet
on PTB. Eq. (13) therefore describes none of the three reported systems, and the
fact that the objective varies by dataset is not disclosed.

Resolution: replace Eq. (13) with the three losses actually implemented and state
which was selected per dataset, or implement the specified loss and re-tune.

---

## Lesser issues

**Matcher differs between the accuracy and security tables.**
`security.per_identity` fuses an identity's enrolment beats into one template by
bit-wise majority vote; Eq. (11) and `protocols.aggregate_scores`, used for the
accuracy tables, average per-template similarities. Majority fusion shifts scores
upward, so a `tau` fixed with the averaging matcher is permissive for the fused
one. Visible in the released data: `security_ecgid_proposed_L512_s0.json` reports
96.8 % genuine acceptance at what should be the FAR = 1 % point, against
`tpr_at_far1 = 70.2` for the same system in `table_main.json`. It also explains
why the control reads 6.85 in `control_security.json` and 6.75 in Table V.

**ECG-ID post-hoc baselines run at L=256 while the proposed system runs at
L=512.** `run_final.eval_job` calls `evaluate.py` without `--L`, and
`fill_tables.py` and `irr.py` hardcode 256. This is the same confound the paper
describes and reports as fixed — it was fixed for the control only, not for
BioHashing, random projection and compressive sensing.

**Transform-known impostor rate includes genuine trials.** `attack_stolen_key`
scores all query embeddings against all enrolment templates and takes the rate
over the whole matrix; the released `n_trials = 50368 = 1574 x 32` on ECG-ID
confirms the full cross-product, so about one trial in 32 is a mated comparison
counted as an attack. The same file's `revocation()` computes the correctly
filtered quantity and reports 30.7 where the attack reports 32.9.

**The template-plus-transform setting is specified but never measured.** Table III
defines four settings and Section III-F itemises an attack for each; `security.py`
implements three. Table VIII carries three ASR columns.

**`control_security.py` uses a different estimator for the guessing rate.** It
draws 200 candidates in total and reports the fraction of attempts that succeed;
every other row reports the fraction of identities compromised under 200 attempts
each. The two are not comparable, and the control's 0.0 / 0.4 / 0.1 are multiples
of 1/1000 rather than of 1/32, 1/69, 1/39 as the other rows are.

**IRR attacker training data is misdescribed.** Section III-F says the classifier
is trained on the training split; `irr.py` trains on the enrolment-side templates
of the evaluation subjects and tests on their query side, which is what the
closed-set framing requires and what `irr.py`'s own docstring says.

**RevScore is a per-seed mean.** All three call sites compute Eq. (2) per seed and
average, which is correct, but the paper does not say so; recomputing Eq. (2) from
the averaged EERs printed in Table VII gives 0.660 for the ECG-ID proposed row
rather than the printed 0.581.

**The leakage assertion runs on record-id sets, not on the materialised arrays**
as Section IV-A and this README state. The check is logically equivalent, since
beats are selected by record membership, but the wording does not describe the
code.

**Minor.** `models.KEY_BITS` is unused and `derive_seed`'s docstring misdescribes
which bits of the digest are taken. `metrics.revscore` is dead code. `aggregate.py`
groups by (dataset, method) only, so re-running it now would pool the L=256 and
L=512 control runs; `verify_tables.py` reads the L=512 files directly to avoid
this. The full-disclosure attack omits the sqrt(d) rescaling in `security.py` but
includes it in `fill_tables.py`, which does not change the reported ASR (the sign
is scale-invariant) but makes `mean_iterations` incomparable between the two.


---

## What was done, and what it changed

**1 — resolved in the paper.** Rather than re-engineer the scheme, the manuscript
now claims what was measured. Section III-C states that the key is derived per
application and version, that all users of one application share a projection,
and that revocation is therefore scoped to an application rather than to a single
user. It also notes that a per-user key is a one-line change to Eq. (1) but is not
free, since the encoder is trained against one particular projection, and that the
stored-template attack is informative *because* the projection is shared — under a
per-user key that attack would be uninformative. No per-user claim survives
anywhere in the paper.

**2 — fixed and re-run.** `derive_seed` defaults to `bind_user=True` while
`CancelableLayer` passes `False`, so the twelve call sites in `security.py`,
`irr.py`, `irr_vs_k.py`, `length_sweep.py` and `fix_fixedproj_link.py` now pass
`bind_user=False` explicitly. The seeds were checked to agree exactly
(2762306544840112576 on both paths for `app-A`, version 0). Every security, IRR
and length measurement was recomputed. The stored-template attack success rates
did not move at all, which is itself informative: that quantity depends on the
geometry of the protected space rather than on which orthonormal projection was
drawn. Linkability and identity recovery did move by a few points, so those
columns were wrong before and are right now.

**3 — disclosed, and measured without it.** The term is now the fourth in
Eq. (12) with its weight in Table II, and Section III-E defines it. It is not
load-bearing: every protected variant was retrained with the weight set to zero
and EER moved by at most 1.05 points with no consistent sign across the three
databases, so no conclusion in the paper rests on distillation being present.

**4 — fixed and retrained.** `train_teacher` now fits one teacher per (database,
seed) on that seed's own split. The sixty protected checkpoints trained against
the shared teacher were quarantined rather than deleted; their metadata is kept
alongside the release for comparison. Every protected variant was retrained and
every dependent measurement recomputed. This was the most consequential of the
five: it moved the proposed method's EER and roughly doubled or halved the
stored-template attack rate depending on the database, and it reversed the
direction of the accuracy comparison on Heartprint. Section IV-G of the paper
quantifies the change.

**5 — resolved in the paper.** Section III-E now gives all three metric losses
that the code implements, and Table II records that the equal-budget search
selected the contrastive form on ECG-ID, the matching-aware form on Heartprint and
the batch-hard triplet on PTB. The paper no longer claims a single objective it
did not optimise.

## Still open

The lesser issues listed above have not been addressed: the security tables still
fuse enrolment beats by majority vote where the accuracy tables average per-beat
similarities, the ECG-ID post-hoc baselines are still measured at L=256 while the
proposed system uses L=512, the transform-known column still includes mated
trials, the template-plus-transform setting is specified but not measured, and
`control_security.py` still estimates the guessing rate over attempts rather than
over identities. Each is documented above with a concrete fix. None of them
affects the direction of any conclusion in the paper, but all of them would have
to be settled before these numbers were used for anything else.
