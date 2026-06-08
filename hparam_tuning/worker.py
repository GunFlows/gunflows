#!/usr/bin/env python3
import uuid, os, sys, subprocess, yaml, optuna, re, datetime, getpass, signal, math, random

STUDY      = os.getenv("OPTUNA_STUDY_NAME",  "hk_study")
STORE      = os.getenv("OPTUNA_STORAGE",     "sqlite:///optuna_hk.db")
BUDGET     = int(os.getenv("TOTAL_TRIALS",   "100"))
EXPERIMENT = os.getenv("OPTUNA_EXPERIMENT",  "demonstrator_100plus10_fakedata")

with open("search_space.yaml") as f:
    SPACE = yaml.safe_load(f)["parameters"]

_current_proc: "subprocess.Popen | None" = None

def _sigterm_handler(signum, frame):
    # Let the subprocess die (it also received SIGTERM from SLURM);
    # do NOT exit here so the readline loop can drain and we can parse the loss.
    if _current_proc is not None:
        _current_proc.terminate()

signal.signal(signal.SIGTERM, _sigterm_handler)

def sug(t, n, c):
    if c["type"] == "float":       return t.suggest_float(n, c["min"], c["max"], log=c.get("log", False))
    if c["type"] == "int":         return t.suggest_int(n, c["min"], c["max"], step=c.get("step", 1), log=c.get("log", False))
    if c["type"] == "categorical": return t.suggest_categorical(n, c["choices"])
    raise ValueError

# Match flexible "Best ... loss" and any "val_loss" assignment
RX_BEST = re.compile(r"Best.*loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE)
RX_VAL  = re.compile(r"val[_\s-]*loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE)

def _extract_loss(text: str):
    # Best (minimum) loss over the whole run -- this is the intended objective.
    # We scan every "Best loss"/"val_loss" line and keep the smallest.
    # (Previously this returned the *last* such line, i.e. the loss at the point
    # the run ended; for a diverged or walltime-killed run that is much worse
    # than the best the model actually reached.)
    best = None
    for ln in text.splitlines():
        m = RX_BEST.search(ln) or RX_VAL.search(ln)
        if m:
            v = float(m.group(1))
            best = v if best is None else min(best, v)
    return best

CMD_BASE = [
    "apptainer","exec","--nv",
    "--pwd","/workspace/work/GuNFlows",
    "--bind","/home/shares/sanchezf/gundam_n_flow/GuNFlows:/workspace/work/GuNFlows",
    "--bind","/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/gunflows_dev",
    "--bind","/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace:/workspace/config",
    "--bind","/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/DATA:/workspace/data",
    "--env","PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows",
    "/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif",
    "bash","-lc"
]

def run(overrides):
    global _current_proc
    cmd = CMD_BASE + [
        "source /workspace/work/GuNFlows/setup_nosubshell.sh && "
        "HYDRA_FULL_ERROR=1 "
        # exec so python REPLACES the bash shell: a SIGTERM from apptainer (sent
        # by _sigterm_handler at SLURM walltime) then reaches the training
        # process directly instead of dying in bash, so the loss can be salvaged.
        "exec python -s -m gunflows.train "
        "--config-path /workspace/work/GuNFlows/hparam_tuning/configs "
        "--config-name config "
        + " ".join(overrides)
    ]

    lines = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    ) as p:
        _current_proc = p
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            sys.stderr.write(line)
            lines.append(line)
        p.wait()
        _current_proc = None
        if p.returncode:
            print(f"[worker] process exited with code {p.returncode} — attempting to parse partial loss", file=sys.stderr)

    return "".join(lines)


def objective(trial):
    now  = datetime.datetime.now()
    user = getpass.getuser()
    uid  = uuid.uuid4().hex[:8]
    run_base = f"hparam_tuning/databases/{STUDY}/runs/{user}"
    outdir   = f"{run_base}/{now:%Y-%m-%d}_{now:%H-%M-%S}_{uid}"
    ov = [f"experiment={EXPERIMENT}"]
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

# ---------------------------------------------------------------------------
# Active exploration: probe the emptiest region of the (fixed) search space.
# TPE has collapsed onto a couple of basins; these helpers let a share of the
# parallel workers sample the point that is maximally far from every trial seen
# so far, so under-explored N-dim regions keep getting covered. Search-space
# bounds are NOT changed -- exploration stays inside them.
# ---------------------------------------------------------------------------
def _unit(c, value):
    """Map a parameter value into [0,1] for distance computations."""
    if c["type"] == "categorical":
        ch = c["choices"]
        return ch.index(value) / max(len(ch) - 1, 1)
    lo, hi = c["min"], c["max"]
    if c.get("log"):
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)

def _draw(c, rng):
    """Draw one uniformly-random value respecting the param's type/scale/bounds."""
    if c["type"] == "categorical":
        return rng.choice(c["choices"])
    lo, hi = c["min"], c["max"]
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if c.get("log") else rng.uniform(lo, hi)
    if c["type"] == "int":
        step = c.get("step", 1)
        v = lo + round((v - lo) / step) * step
        return int(min(max(v, lo), hi))
    return v

def _farthest_point(study, rng, n_candidates=4000, top_k=8):
    """Params for a point in an empty region: of `n_candidates` random points,
    keep the `top_k` with the largest min-distance (unit cube) to all existing
    trials and pick one at random, so parallel explorers diverge."""
    pts = []
    for t in study.trials:
        if not t.params:
            continue
        try:
            pts.append([_unit(c, t.params[k]) for k, c in SPACE.items()])
        except (KeyError, ValueError):
            continue
    scored = []
    for _ in range(n_candidates):
        cand = {k: _draw(c, rng) for k, c in SPACE.items()}
        u = [_unit(c, cand[k]) for k, c in SPACE.items()]
        d = 1.0 if not pts else min(sum((a - b) ** 2 for a, b in zip(u, p)) for p in pts)
        scored.append((d, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return rng.choice([c for _, c in scored[:top_k]])

def main():
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    job_id  = int(os.getenv("SLURM_ARRAY_JOB_ID", os.getenv("SLURM_JOB_ID", "0")))
    seed = (job_id * 131 + task_id * 17) & 0x7fffffff

    # Seeded multivariate TPE: distinct seeds de-correlate the parallel workers
    # (they no longer all collapse onto the same suggestion), consider_endpoints
    # lets it probe the domain edges.
    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, group=True, consider_endpoints=True,
    )
    try:
        study = optuna.load_study(study_name=STUDY, storage=STORE, sampler=sampler)
    except KeyError:
        study = optuna.create_study(study_name=STUDY, storage=STORE,
                                    direction="minimize", sampler=sampler)

    # EXPLORE_WORKERS of every 5 workers explore the emptiest region instead of
    # exploiting (default 2 -> 40% exploration, 60% TPE). Set to 0 to disable.
    n_explore = int(os.getenv("EXPLORE_WORKERS", "2"))
    if task_id % 5 < n_explore:
        params = _farthest_point(study, random.Random(seed))
        study.enqueue_trial(params, user_attrs={"explore": True}, skip_if_exists=False)
        print(f"[worker] EXPLORE (farthest-point) enqueued: {params}", file=sys.stderr)

    study.optimize(objective, n_trials=BUDGET, gc_after_trial=True)

if __name__ == "__main__":
    main()