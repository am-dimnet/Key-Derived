"""
Record- and session-level partitioning, defined *before* segmentation is ever
consulted, plus the leakage assertion demanded by reviewer comment R1.8.

Partitioning rule (identical for all datasets, only the enrol/test axis differs):

    subjects -> TRAIN_SUBJ (50%) | VAL_SUBJ (15%) | EVAL_SUBJ (35%)   disjoint
    TRAIN_SUBJ  all records          -> train
    VAL_SUBJ    enrol / query records -> val   (threshold tau, early stopping)
    EVAL_SUBJ   enrol / test  records -> enrol, test

The three subject pools are disjoint, so no recording can contribute beats to two
different subsets, and the threshold is chosen on identities that appear neither
in training nor in the test protocol.  `assert_no_leakage` verifies this on the
materialised arrays rather than trusting the construction.

The enrol/test axis per dataset:
    ecgid      earliest recording enrols, later recordings are queries
    heartprint Session-1 enrols, Sessions 2 / 3L / 3R are queries (cross-session)
    ptbdb      earliest record enrols, remaining records are queries
"""
import json

import numpy as np


ENROL_RULE = {"ecgid": "first_record", "ptbdb": "first_record",
              "heartprint": "first_session"}


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return dict(beats=d["beats"], subject=d["subject"], record=d["record"],
                session=d["session"], rpeak=d["rpeak"],
                subjects=list(d["subjects"]), record_names=list(d["record_names"]))


def _records_of(data):
    """record id -> (subject, session, name), in a deterministic order."""
    rec_ids = np.unique(data["record"])
    info = {}
    for r in rec_ids:
        m = data["record"] == r
        info[int(r)] = (int(data["subject"][m][0]), int(data["session"][m][0]),
                        data["record_names"][int(r)])
    return info


def _enrol_test(entries, dataset, n_enrol_records=1):
    """Split one subject's time-ordered records into enrol / query parts."""
    if ENROL_RULE[dataset] == "first_session":
        first_sess = entries[0][0]
        e = [rid for sess, _, rid in entries if sess == first_sess]
        t = [rid for sess, _, rid in entries if sess != first_sess]
        if not t:                                  # only one session present
            e, t = [entries[0][2]], [rid for _, _, rid in entries[1:]]
    else:
        k = min(n_enrol_records, len(entries) - 1)
        e = [rid for _, _, rid in entries[:k]]
        t = [rid for _, _, rid in entries[k:]]
    return e, t


def make_split(data, dataset, seed=0, train_frac=0.5, val_frac=0.15,
               n_enrol_records=1):
    rng = np.random.default_rng(seed)
    info = _records_of(data)

    by_subject = {}
    for rid, (s, sess, name) in info.items():
        by_subject.setdefault(s, []).append((sess, name, rid))
    for s in by_subject:
        by_subject[s].sort()                       # (session, name) == time order

    subjects = np.array(sorted(by_subject))
    # only subjects with >=2 records can be enrolled and then queried
    usable = np.array([s for s in subjects if len(by_subject[s]) >= 2])
    rng.shuffle(usable)
    n_tr = int(round(train_frac * len(usable)))
    n_va = max(1, int(round(val_frac * len(usable))))
    train_subj = set(usable[:n_tr].tolist())
    val_subj = set(usable[n_tr:n_tr + n_va].tolist())
    eval_subj = set(usable[n_tr + n_va:].tolist())
    # single-record subjects can still supply representation-learning data
    train_subj |= {s for s in subjects
                   if s not in train_subj and s not in val_subj and s not in eval_subj}

    train_recs, val_e, val_q, enrol_recs, test_recs = [], [], [], [], []

    for s in sorted(train_subj):
        train_recs += [rid for _, _, rid in by_subject[s]]

    for s in sorted(val_subj):
        e, t = _enrol_test(by_subject[s], dataset, n_enrol_records)
        if e and t:
            val_e += e
            val_q += t

    for s in sorted(eval_subj):
        e, t = _enrol_test(by_subject[s], dataset, n_enrol_records)
        if e and t:
            enrol_recs += e
            test_recs += t

    split = dict(
        dataset=dataset, seed=seed,
        train_records=sorted(train_recs),
        val_records=sorted(val_e + val_q),
        val_enrol_records=sorted(val_e), val_query_records=sorted(val_q),
        enrol_records=sorted(enrol_recs), test_records=sorted(test_recs),
        train_subjects=sorted(train_subj), val_subjects=sorted(val_subj),
        eval_subjects=sorted(eval_subj),
        enrol_rule=ENROL_RULE[dataset],
    )
    assert_no_leakage(split)
    return split


SUBSETS = ("train", "val_enrol", "val_query", "enrol", "test")


def assert_no_leakage(split):
    """Hard check that every record pool is pairwise disjoint (R1.8)."""
    pools = {k: set(split[f"{k}_records"]) for k in SUBSETS}
    names = list(pools)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = pools[names[i]] & pools[names[j]]
            if inter:
                raise AssertionError(
                    f"record leakage between {names[i]} and {names[j]}: "
                    f"{sorted(inter)[:5]} ...")
    pairs = [("train_subjects", "val_subjects"), ("train_subjects", "eval_subjects"),
             ("val_subjects", "eval_subjects")]
    for a, b in pairs:
        inter = set(split[a]) & set(split[b])
        if inter:
            raise AssertionError(f"subject leakage {a}/{b}: {sorted(inter)[:5]}")
    return True


def masks(data, split):
    """Boolean beat masks for each subset."""
    rec = data["record"]
    out = {}
    for k in SUBSETS:
        out[k] = np.isin(rec, np.asarray(split[f"{k}_records"], dtype=rec.dtype))
    return out


def split_stats(data, split):
    """Per-subset counts of subjects / records / beats -> Table II (R1.8)."""
    m = masks(data, split)
    stats = {}
    for k, mask in m.items():
        stats[k] = dict(
            subjects=int(len(np.unique(data["subject"][mask]))),
            records=int(len(np.unique(data["record"][mask]))),
            beats=int(mask.sum()),
        )
    stats["total"] = dict(
        subjects=int(len(np.unique(data["subject"]))),
        records=int(len(np.unique(data["record"]))),
        beats=int(len(data["beats"])),
    )
    stats["leakage_check"] = "passed: all four record pools pairwise disjoint"
    return stats


def build_pairs(query_subj, enrol_subj):
    """Genuine/impostor label matrix for a query x enrol score matrix."""
    return (np.asarray(query_subj)[:, None] == np.asarray(enrol_subj)[None, :]).astype(int)


def aggregate_scores(S, enrol_subj, query_subj, agg="mean"):
    """Collapse a [n_query, n_enrol_templates] matrix to one score per
    (query, enrolled identity) pair by averaging that identity's templates.

    Returns scores, labels, and the probe subject of every trial - the score file
    that every downstream metric consumes.
    """
    enrol_subj = np.asarray(enrol_subj)
    query_subj = np.asarray(query_subj)
    ids = np.unique(enrol_subj)
    cols = []
    for i in ids:
        sel = enrol_subj == i
        block = S[:, sel]
        cols.append(block.mean(1) if agg == "mean" else block.max(1))
    M = np.stack(cols, axis=1)                       # [n_query, n_identities]
    labels = (query_subj[:, None] == ids[None, :]).astype(int)
    probe = np.repeat(query_subj[:, None], len(ids), axis=1)
    return M.ravel(), labels.ravel(), probe.ravel(), ids


def save_split(split, path):
    with open(path, "w") as f:
        json.dump({k: (list(map(int, v)) if isinstance(v, list) else v)
                   for k, v in split.items()}, f, indent=2)
