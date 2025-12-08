#!/usr/bin/env python3
import uuid, os, sys, subprocess, yaml, optuna, re, datetime, getpass

STUDY  = os.getenv("OPTUNA_STUDY_NAME", "hk_study")
STORE  = os.getenv("OPTUNA_STORAGE",   "sqlite:///optuna_hk.db")
BUDGET = int(os.getenv("TOTAL_TRIALS", "100"))

with open("search_space.yaml") as f:
    SPACE = yaml.safe_load(f)["parameters"]

def sug(t, n, c):
    if c["type"] == "float":       return t.suggest_float(n, c["min"], c["max"], log=c.get("log", False))
    if c["type"] == "int":         return t.suggest_int(n, c["min"], c["max"], step=c.get("step", 1), log=c.get("log", False))
    if c["type"] == "categorical": return t.suggest_categorical(n, c["choices"])
    raise ValueError

# Match flexible "Best ... loss" and any "val_loss" assignment
RX_BEST = re.compile(r"Best.*loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE)
RX_VAL  = re.compile(r"val[_\s-]*loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE)

def _extract_loss(text: str):
    last_val = None
    for ln in reversed(text.splitlines()):
        m_best = RX_BEST.search(ln)
        if m_best:
            return float(m_best.group(1))
        m_val = RX_VAL.search(ln)
        if m_val and last_val is None:
            last_val = float(m_val.group(1))
    return last_val

CMD_BASE = [
    "apptainer","exec","--nv",
    "--pwd","/workspace/work/GuNFlows",
    "--bind","/srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud:/workspace/data",
    "--bind","/home/shares/sanchezf/gundam_n_flow/GuNFlows:/workspace/work/GuNFlows",
    "--bind","/srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config",
    "--env","DATA_DIR=/workspace/data/DatasetNFlowsOA2022/Asimov/allParameters",
    "--env","PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows",
    "/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif",
    "bash","-lc" 
]

def run(overrides):
    cmd = CMD_BASE + [
        "source /workspace/work/GuNFlows/setup_nosubshell.sh && "
        "HYDRA_FULL_ERROR=1 "
        "python -s -m gunflows.train "
        "--config-path /workspace/work/GuNFlows/hparam_tuning/configs "
        "--config-name config "
        + " ".join(overrides)
    ]

    lines = []
    # Note: no text=True here, we work with raw bytes
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    ) as p:
        for raw in iter(p.stdout.readline, b""):
            # Decode robustly: replace invalid sequences instead of crashing
            line = raw.decode("utf-8", errors="replace")
            sys.stderr.write(line)
            lines.append(line)
        p.wait()
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, cmd)

    return "".join(lines)


def objective(trial):
    now  = datetime.datetime.now()
    user = getpass.getuser()
    uid  = uuid.uuid4().hex[:8]
    run_base = f"hparam_tuning/databases/{STUDY}/runs/{user}"
    outdir   = f"{run_base}/{now:%Y-%m-%d}_{now:%H-%M-%S}_{uid}"
    ov = ["experiment=oa2022_data"]
    ov += [f"{k}={sug(trial, k, c)}" for k, c in SPACE.items()]
    ov.append(f"hydra.run.dir={outdir}")
    out = run(ov)
    debug_tail = "\n".join(out.splitlines()[-40:])
    print("\n===== WORKER CAPTURED OUTPUT (tail) =====", file=sys.stderr)
    print(debug_tail, file=sys.stderr)
    print("=========================================\n", file=sys.stderr)
    loss = _extract_loss(out)
    print(f"[worker] parsed loss: {loss}", file=sys.stderr)
    if loss is not None:
        return loss
    raise optuna.exceptions.TrialPruned("Validation loss not found in logs")

def main():
    try:
        study = optuna.load_study(study_name=STUDY, storage=STORE)
    except KeyError:
        study = optuna.create_study(study_name=STUDY, storage=STORE, direction="minimize")

    study.optimize(objective, n_trials=BUDGET, gc_after_trial=True)

if __name__ == "__main__":
    main()