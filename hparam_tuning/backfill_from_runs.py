#!/usr/bin/env python3
"""Recover trials that were lost when SLURM killed workers at walltime before
_cleanup could copy their DB into tmp_dbs (so merge_stage found nothing).

Each trial left a hydra run dir under databases/<study>/runs/<user>/ containing:
  - .hydra/overrides.yaml : the exact optuna-sampled params
  - stage_log.csv         : per-epoch val_loss (min == the "Best loss" the
                            worker would have parsed and returned)

We reconstruct those trials and insert any that are not already in the master
study, deduping by parameter fingerprint. Run with --apply to write; default is
a dry run.
"""
import csv
import glob
import os
import re
import sys
import shutil
import datetime
import subprocess
import yaml
import optuna
from optuna.distributions import FloatDistribution, IntDistribution

STUDY = "hp_fakedata_paper"
BASE = "/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning"
RUNS = f"{BASE}/databases/{STUDY}/runs"
MASTER = f"{BASE}/databases/{STUDY}/{STUDY}.db"
MASTER_URI = f"sqlite:///{MASTER}"
APPLY = "--apply" in sys.argv

# A run dir started after a still-running worker began is a live trial whose
# loss is not final yet — exclude it. Cutoff = earliest start of any running
# gunflows_optuna job (minus a small margin), parsed from squeue.
def live_cutoff():
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "-h", "-n", "gunflows_optuna", "-O", "StartTime"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    starts = []
    for ln in out.split():
        try:
            starts.append(datetime.datetime.strptime(ln.strip(), "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            pass
    if not starts:
        return None
    return min(starts) - datetime.timedelta(minutes=5)

CUTOFF = live_cutoff()
if CUTOFF:
    print(f"excluding run dirs started at/after {CUTOFF:%Y-%m-%d %H:%M:%S} (live jobs)")

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")
def run_started(name):
    m = _TS_RE.search(name)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1) + "_" + m.group(2), "%Y-%m-%d_%H-%M-%S")

# Build distributions straight from the search space so stored params/dedup
# match exactly what suggest_* produced.
with open(f"{BASE}/search_space.yaml") as f:
    SPACE = yaml.safe_load(f)["parameters"]

def make_dist(c):
    if c["type"] == "float":
        return FloatDistribution(float(c["min"]), float(c["max"]), log=c.get("log", False))
    if c["type"] == "int":
        return IntDistribution(int(c["min"]), int(c["max"]), step=int(c.get("step", 1)), log=c.get("log", False))
    raise ValueError(c["type"])

DISTS = {k: make_dist(c) for k, c in SPACE.items()}

def parse_overrides(path):
    out = {}
    for item in yaml.safe_load(open(path)):
        k, _, v = item.partition("=")
        if k in DISTS:
            out[k] = int(v) if isinstance(DISTS[k], IntDistribution) else float(v)
    return out

def best_loss(stage_log):
    # The intended objective is the BEST (minimum) val_loss over the run.
    # NOTE: worker.py's _extract_loss currently returns the *last* "Best loss"
    # line instead (a bug), so natively recorded trials may hold a worse value.
    best = None
    with open(stage_log) as f:
        for row in csv.DictReader(f):
            v = row.get("val_loss", "").strip()
            if v:
                fv = float(v)
                best = fv if best is None else min(best, fv)
    return best

def fingerprint(params):
    return (params["experiment.model.nflows"],
            params["experiment.model.nbins"],
            round(params["experiment.optim.lr"], 15),
            round(params["experiment.model.tail_bound"], 10))

study = optuna.load_study(study_name=STUDY, storage=MASTER_URI)
storage = study._storage
existing = {fingerprint(t.params) for t in study.trials if t.params}
print(f"master DB currently has {len(study.trials)} trials")

run_dirs = sorted(glob.glob(f"{RUNS}/*/*/"))
now = datetime.datetime.now().timestamp()

to_add, skipped = [], []
for d in run_dirs:
    ov = os.path.join(d, ".hydra", "overrides.yaml")
    sl = os.path.join(d, "stage_log.csv")
    name = os.path.relpath(d, RUNS)
    if not (os.path.exists(ov) and os.path.exists(sl)):
        skipped.append((name, "missing overrides/stage_log")); continue
    started = run_started(name)
    if CUTOFF and started and started >= CUTOFF:
        skipped.append((name, f"live job (started {started:%Y-%m-%d %H:%M})")); continue
    params = parse_overrides(ov)
    if set(params) != set(DISTS):
        skipped.append((name, f"params incomplete: {sorted(params)}")); continue
    loss = best_loss(sl)
    if loss is None:
        skipped.append((name, "no val_loss recorded")); continue
    if fingerprint(params) in existing:
        skipped.append((name, f"already in study (loss={loss:.5g})")); continue
    existing.add(fingerprint(params))
    to_add.append((name, params, loss))

print(f"\n=== {len(to_add)} trials to ADD ===")
for name, params, loss in to_add:
    print(f"  + {name}  loss={loss:.6g}  lr={params['experiment.optim.lr']:.4g} "
          f"nflows={params['experiment.model.nflows']} "
          f"nbins={params['experiment.model.nbins']} "
          f"tb={params['experiment.model.tail_bound']:.4g}")

print(f"\n=== {len(skipped)} skipped ===")
for name, why in skipped:
    print(f"  - {name}: {why}")

if not APPLY:
    print("\nDRY RUN — re-run with --apply to write these into the master DB.")
    sys.exit(0)

if not to_add:
    print("nothing to add.")
    sys.exit(0)

bak = f"{MASTER}.bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(MASTER, bak)
print(f"\nbacked up master DB -> {bak}")

for name, params, loss in to_add:
    tid = storage.create_new_trial(study._study_id)
    for pname, dist in DISTS.items():
        storage.set_trial_param(tid, pname, dist.to_internal_repr(params[pname]), dist)
    storage.set_trial_user_attr(tid, "recovered_from", name)
    storage.set_trial_user_attr(tid, "recovered_partial", True)
    storage.set_trial_state_values(tid, optuna.trial.TrialState.COMPLETE, [loss])

study2 = optuna.load_study(study_name=STUDY, storage=MASTER_URI)
print(f"\napplied. master DB now has {len(study2.trials)} trials; "
      f"best value = {study2.best_value:.6g}")
