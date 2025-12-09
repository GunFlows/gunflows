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
        sig = (tuple(sorted(t.params.items())), tuple(t.values or ()))

        if sig in signature:
            continue
        signature.add(sig)        

        new_id = storage.create_new_trial(dst._study_id)

        for pname, dist in t.distributions.items():
            internal = dist.to_internal_repr(t.params[pname])
            storage.set_trial_param(new_id, pname, internal, dist)

        storage.set_trial_state_values(new_id, t.state, t.values)

        for k, v in t.user_attrs.items():
            storage.set_trial_user_attr(new_id, k, v)
        for k, v in t.system_attrs.items():
            storage.set_trial_system_attr(new_id, k, v)

    os.remove(db)

print(f"stage {stage} merged, total unique trials {len(dst.trials)}")
os.system(
    f"python diagnostics.py {study_name} {master_path} {stage}"
)