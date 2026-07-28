"""Single place where the project root is resolved.

The experiments were run on a Linux server with a fixed layout.  For the code
release every path is derived from one root instead, so the package runs
anywhere:

    export ARTICLE4_ROOT=/somewhere/with/space      # optional
    python code/prep.py --dataset ecgid

If ARTICLE4_ROOT is unset the root is the directory containing this package,
so a fresh clone works with no configuration.  The directories below are
created on demand by the scripts that write into them.

    $ARTICLE4_ROOT/data/raw     downloaded databases
    $ARTICLE4_ROOT/data/proc    preprocessed .npz written by prep.py
    $ARTICLE4_ROOT/runs         training checkpoints
    $ARTICLE4_ROOT/results      per-run JSON, score arrays, aggregated tables
    $ARTICLE4_ROOT/logs         stdout of long jobs
"""
import os

ROOT = os.environ.get(
    "ARTICLE4_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

DATA = os.path.join(ROOT, "data")
PROC = os.path.join(DATA, "proc")
RAW = os.path.join(DATA, "raw")
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
LOGS = os.path.join(ROOT, "logs")

for _d in (DATA, PROC, RAW, RUNS, RESULTS, TABLES, LOGS):
    os.makedirs(_d, exist_ok=True)
