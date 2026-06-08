#!/usr/bin/env python3
import glob
import optuna
import os
import sys

study_name = sys.argv[1]
stage      = sys.argv[2]

base_root   = "/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/databases"
base        = f"{base_root}/{study_name}"
tmpdir      = f"{base}/tmp_dbs"
master_path = f"{base}/{study_name}.db"

stage_dbs   = glob.glob(f"{tmpdir}/{study_name}_{stage}_*.db")

if not stage_dbs:
    # No per-worker DB landed in tmp_dbs for this stage. This is the failure
    # mode where workers were killed (e.g. SLURM walltime) before _cleanup
    # could copy their DB, so there is nothing to merge and the master DB
    # silently stops growing. Make it loud instead of exiting 0.
    print(
        f"[merge_stage] WARNING: stage {stage} produced NO databases in "
        f"{tmpdir} (pattern {study_name}_{stage}_*.db). Nothing merged — "
        f"the master DB will not grow for this stage. Likely the workers were "
        f"killed before cleanup (walltime/SIGKILL).",
        file=sys.stderr,
    )
    sys.exit(0)

master_uri = f"sqlite:///{master_path}"
if not os.path.exists(master_path):
    optuna.create_study(study_name=study_name,
                        storage=master_uri,
                        direction="minimize")

dst = optuna.load_study(study_name=study_name, storage=master_uri)
storage = dst._storage           

signature = {
    (tuple(sorted(t.params.items())),
     tuple(t.values or ()))
    for t in dst.trials
}
# 

for db in stage_dbs:
    src = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")

    for t in src.get_trials(deepcopy=False):
        # Only merge finished trials. A worker that was killed at walltime
        # before salvaging leaves its trial in RUNNING state (value=None);
        # merging that would pollute the master DB with None-valued trials.
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        sig = (tuple(sorted(t.params.items())), tuple(t.values or ()))

        if sig in signature:
            continue
        signature.add(sig)        

        new_id = storage.create_new_trial(dst._study_id)

        for pname, dist in t.distributions.items():
            internal = dist.to_internal_repr(t.params[pname])
            storage.set_trial_param(new_id, pname, internal, dist)

        # Attrs MUST be set while the trial is still RUNNING: Optuna forbids
        # updating a finished trial. (Trials now carry attrs -- e.g. the
        # "explore" tag and enqueue_trial's "fixed_params" -- so completing the
        # trial first crashes the merge, which previously killed the whole loop.)
        for k, v in t.user_attrs.items():
            storage.set_trial_user_attr(new_id, k, v)
        for k, v in t.system_attrs.items():
            storage.set_trial_system_attr(new_id, k, v)

        storage.set_trial_state_values(new_id, t.state, t.values)

    os.remove(db)

print(f"stage {stage} merged from stage DBs, unique trials so far {len(dst.trials)}")

# The in-process walltime salvage is unreliable: a worker killed at SLURM
# walltime often leaves its trial RUNNING/value-less, so the stage-DB merge
# above misses it. Every trial nonetheless writes a per-epoch stage_log.csv in
# its run dir, so backfill straight from those to guarantee no completed
# training is lost. Idempotent (dedupes by params) and skips still-live jobs.
os.system("python backfill_from_runs.py --apply")

os.system(
    f"python diagnostics.py {study_name} {master_path} {stage}"
)