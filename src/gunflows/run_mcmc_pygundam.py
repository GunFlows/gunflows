#!/usr/bin/env python3
"""
run_mcmc_pygundam.py — parallel MCMC using the GuNFlows LikelihoodSampler.

Two modes (set via config ``mode``):

  parallel_tempering  (default)
                N workers at temperatures β₁=1 > β₂ > … > βₙ run in parallel.
                Hot chains explore; the cold chain (β=1) samples the target.
                Swaps attempted every swap_interval steps.

  independent   N chains run fully in parallel with different seeds, then merged.
                Zero communication; wall-clock ≈ 1 chain.

Both modes use Cholesky-preconditioned proposals:
    x' = x + σ · L @ z,   z ~ N(0,I),   L = chol(postfit_cov)
with per-chain Robbins-Monro scale adaptation (targeting 23.4% acceptance)
during burn-in, then a frozen scale for production.

Corner plots are saved every plot_every steps (parameter space, KS-ranked
most non-Gaussian dims, MCMC histogram vs postfit Gaussian overlay).

SLURM:  cpus-per-task = n_chains * threads_per_chain
"""
from __future__ import annotations

import os
import multiprocessing as mp
from contextlib import contextmanager

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.special import erf
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from gunflows.likelihood_sampler import LikelihoodSampler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MCMC_COLOR  = "#2ca02c"
GAUSS_COLOR = "#d62728"


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _make_sampler(llh_cfg) -> LikelihoodSampler:
    cwd = llh_cfg.get("cwd", None)
    with _pushd(cwd or os.path.dirname(str(llh_cfg.config))):
        return LikelihoodSampler(
            config_file=str(llh_cfg.config),
            override_files=list(llh_cfg.get("overrides", []) or []),
            data_is_asimov=bool(llh_cfg.get("asimov", False)),
            threads=int(llh_cfg.get("threads_per_chain", 1)),
        )


def _propose(L: np.ndarray, current: np.ndarray, scale: float,
             rng: np.random.Generator) -> np.ndarray:
    return current + scale * (L @ rng.standard_normal(len(current)))


def _rm_update(log_s: float, step: int, accepted: bool, target: float,
               rate: float, lag: float, exp_: float) -> float:
    gamma = rate / (step + lag) ** exp_
    return log_s + gamma * (float(accepted) - target)


def _geom_temperatures(n: int, beta_min: float) -> list[float]:
    if n == 1:
        return [1.0]
    ratio = (beta_min / 1.0) ** (1.0 / (n - 1))
    return [ratio ** i for i in range(n)]


def _short_name(s: str) -> str:
    """Strip long prefix paths from parameter names for plot labels."""
    return s.split("/")[-1] if "/" in s else s


def _resolve_out(cfg: DictConfig, base_dir: str) -> str:
    out_parent = str(cfg.save_dir) if cfg.get("save_dir", None) else base_dir
    os.makedirs(out_parent, exist_ok=True)
    return os.path.join(out_parent, str(cfg.out_file))


# ─────────────────────────────────────────────────────────────────────────────
# Corner plot  (style mirrors make_paper_plots.plot_param_corner)
# ─────────────────────────────────────────────────────────────────────────────

def _ks_score(z_col: np.ndarray) -> float:
    ok = np.isfinite(z_col)
    if not ok.any():
        return 0.0
    zs = np.sort(z_col[ok])
    n  = len(zs)
    cdf_emp  = np.arange(1, n + 1) / n
    cdf_norm = 0.5 * (1.0 + erf(zs / np.sqrt(2.0)))
    return float(np.max(np.abs(cdf_emp - cdf_norm)))


def plot_corner(
    chain: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    par_names: list[str],
    step: int,
    out_path: str,
    n_params: int = 10,
    bins: int = 50,
    xsec_only: bool = False,
) -> None:
    """
    Parameter corner plot.
    If xsec_only=True, restricts to cross-section (Non-Linear) parameters and
    shows all of them (KS-ranked within that subset).
    Otherwise shows the top n_params dims by KS score across all parameters.
    Diagonal  : MCMC histogram (green) + postfit Gaussian (dashed red).
    Off-diag  : 2D histogram of MCMC samples (log colour scale).
    Saved to out_path (png, overwritten each call).
    """
    std = np.sqrt(np.clip(np.diag(cov), 1e-14, None))
    z   = (chain - mean) / std

    ks  = np.array([_ks_score(z[:, i]) for i in range(z.shape[1])])

    if xsec_only:
        # keep only Non-Linear (cross-section spline) parameters, then KS-rank
        xsec_idx = [i for i, name in enumerate(par_names) if "Non-Linear" in str(name)]
        if not xsec_idx:
            print("  WARNING: no Non-Linear parameters found, falling back to all params", flush=True)
            xsec_idx = list(range(len(par_names)))
        ks_xsec = ks[xsec_idx]
        sel = [xsec_idx[i] for i in np.argsort(-ks_xsec)]
    else:
        sel = list(np.argsort(-ks)[:n_params])
    n   = len(sel)

    ss   = chain[:, sel]
    ms   = mean[sel]
    sts  = std[sel]
    labs = [_short_name(par_names[i]) if i < len(par_names) else f"par {i}"
            for i in sel]

    cell = 2.8
    fig, axes = plt.subplots(n, n, figsize=(n * cell, n * cell))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue

            if row == col:
                x  = ss[:, row]
                ok = np.isfinite(x)
                rng_lo = ms[row] - 4 * sts[row]
                rng_hi = ms[row] + 4 * sts[row]
                ax.hist(x[ok], bins=bins, range=(rng_lo, rng_hi), density=True,
                        histtype="stepfilled", alpha=0.4, color=MCMC_COLOR)
                ax.hist(x[ok], bins=bins, range=(rng_lo, rng_hi), density=True,
                        histtype="step", linewidth=1.2, color=MCMC_COLOR)
                xs = np.linspace(rng_lo, rng_hi, 300)
                ax.plot(xs,
                        np.exp(-0.5 * ((xs - ms[row]) / sts[row]) ** 2)
                        / (sts[row] * np.sqrt(2 * np.pi)),
                        color=GAUSS_COLOR, linewidth=1.3, linestyle="--")
            else:
                x, y = ss[:, col], ss[:, row]
                ok   = np.isfinite(x) & np.isfinite(y)
                ax.hist2d(x[ok], y[ok], bins=40, norm=LogNorm(), cmap="viridis")

            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(which="both", top=False, right=False, length=4, width=1.0)
            if row < n - 1:
                ax.tick_params(labelbottom=False, bottom=True)
            else:
                ax.set_xlabel(labs[col], fontsize=8, labelpad=6)
                ax.tick_params(labelsize=7)
            if col == 0 and row > 0:
                ax.set_ylabel(labs[row], fontsize=8, labelpad=6)
                ax.tick_params(labelsize=7)
            else:
                ax.tick_params(labelleft=False, left=True)

    subset_label = "xsec (Non-Linear) params" if xsec_only else f"top {n} dims by KS"
    fig.suptitle(
        f"MCMC vs postfit Gaussian — {step:,} steps  "
        f"({subset_label},  green=MCMC  red--=Gaussian)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  corner → {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Worker (one process per temperature / independent chain)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(rank: int,
            cfg_dict: dict,
            beta: float,
            seed: int,
            out_path: str,
            conn):          # mp.Connection or None (independent mode)
    """
    Owns one LikelihoodSampler.

    Protocol (conn is None for independent mode):
      Main → Worker:  ("step_burn", n, rng_seed)
                      ("step_prod", n, rng_seed)
                      ("swap", state, nll)
                      ("quit",)
      Worker → Main:  ("ready", state, nll, par_names, bestfit_nll, L, mean)
                      ("done",  state, nll, n_acc [, chain, nlls])
                      ("swap_ack", rank)
                      ("quit_ack", rank)
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_dict)

    n_thr = int(cfg.likelihood.get("threads_per_chain", 1))
    os.environ["OMP_NUM_THREADS"] = str(n_thr)

    sampler = _make_sampler(cfg.likelihood)

    current   = np.asarray(sampler.postfit_parameter_values, dtype=float)
    mean_init = current.copy()          # best-fit = Gaussian mean
    current_nll, _, _ = sampler.inject_params_and_compute_likelihood(
        current, extend_continue=False)
    if current_nll == -1:
        raise RuntimeError(f"[rank {rank}] best-fit outside domain.")

    cov = np.asarray(sampler.postfit_covariance_matrix, dtype=float)
    L   = np.linalg.cholesky(cov)
    par_names   = sampler.get_parameter_names()
    bestfit_nll = float(sampler.likelihood_at_bestfit)

    if conn is not None:
        conn.send(("ready", current.copy(), current_nll,
                   par_names, bestfit_nll, L, mean_init))
    else:
        print(f"[rank {rank}] ready  dim={len(current)}  NLL={current_nll:.4f}",
              flush=True)

    target_acc = float(cfg.get("target_acceptance", 0.234))
    rm_rate    = float(cfg.get("adapt_rate",     1.0))
    rm_lag     = float(cfg.get("adapt_lag",    100.0))
    rm_exp     = float(cfg.get("adapt_exponent", 0.6))
    burn_in    = int(cfg.burn_in)
    n_steps    = int(cfg.n_steps)
    plot_every = int(cfg.get("plot_every", 10000))
    log_scale  = np.log(float(cfg.get("initial_scale", 1.0)))

    def run_steps(n: int, rng: np.random.Generator, phase: str, step_offset: int):
        nonlocal current, current_nll, log_scale
        chain_, nlls_, n_acc_ = [], [], 0
        for k in range(n):
            scale    = np.exp(log_scale)
            proposal = _propose(L, current, scale, rng)
            nll_p, _, _ = sampler.inject_params_and_compute_likelihood(
                proposal, extend_continue=False)
            accepted = False
            if nll_p != -1:
                if rng.random() < np.exp(min(0.0, beta * (current_nll - nll_p))):
                    current     = proposal
                    current_nll = nll_p
                    accepted    = True
            if phase == "burn":
                log_scale = _rm_update(log_scale, step_offset + k, accepted,
                                       target_acc, rm_rate, rm_lag, rm_exp)
            chain_.append(current.copy())
            nlls_.append(current_nll)
            n_acc_ += int(accepted)
        return chain_, nlls_, n_acc_

    # ── Independent mode: run full chain here, save periodic corner plots ────
    if conn is None:
        out_dir = os.path.dirname(out_path)
        plots_dir = os.path.join(out_dir, f"plots_rank{rank:02d}")
        os.makedirs(plots_dir, exist_ok=True)

        rng = np.random.default_rng(seed)

        if burn_in > 0:
            _, _, n_acc_b = run_steps(burn_in, rng, "burn", 0)
            print(f"[rank {rank}] burn-in done  acc={n_acc_b/burn_in:.3f}"
                  f"  scale={np.exp(log_scale):.4f}", flush=True)

        chain_arr = np.zeros((n_steps, len(current)))
        nlls_arr  = np.zeros(n_steps)
        n_acc_p   = 0
        step = 0
        while step < n_steps:
            batch = min(plot_every, n_steps - step)
            c_, n_, a_ = run_steps(batch, rng, "prod", 0)
            chain_arr[step:step+batch] = c_
            nlls_arr[step:step+batch]  = n_
            n_acc_p += a_
            step    += batch
            print(f"[rank {rank}] prod {step}/{n_steps}"
                  f"  NLL={current_nll:.4f}  acc={n_acc_p/step:.3f}", flush=True)
            plot_corner(
                chain_arr[:step], mean_init, cov, par_names, step,
                os.path.join(plots_dir, f"corner_{step:07d}.png"),
                xsec_only=bool(cfg_dict.get("xsec_only", False)),
            )

        np.savez(out_path, chain=chain_arr, nll=nlls_arr,
                 par_names=np.array(par_names), bestfit_nll=bestfit_nll,
                 final_scale=float(np.exp(log_scale)),
                 acceptance_rate=float(n_acc_p / n_steps))
        print(f"[rank {rank}] saved → {out_path}", flush=True)
        return

    # ── PT mode: event-driven via conn ──────────────────────────────────────
    step_burn = 0
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            conn.send(("quit_ack", rank))
            break
        elif msg[0] == "step_burn":
            n, seed_step = msg[1], msg[2]
            c_, n_, n_acc = run_steps(n, np.random.default_rng(seed_step), "burn", step_burn)
            step_burn += n
            conn.send(("done", current.copy(), current_nll, n_acc))
        elif msg[0] == "step_prod":
            n, seed_step = msg[1], msg[2]
            c_, n_, n_acc = run_steps(n, np.random.default_rng(seed_step), "prod", 0)
            conn.send(("done", current.copy(), current_nll, n_acc, c_, n_))
        elif msg[0] == "swap":
            current     = np.asarray(msg[1], dtype=float)
            current_nll = float(msg[2])
            conn.send(("swap_ack", rank))


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Tempering coordinator
# ─────────────────────────────────────────────────────────────────────────────

def _run_pt(cfg: DictConfig, base_dir: str) -> None:
    n_chains   = int(cfg.get("n_chains", 4))
    beta_min   = float(cfg.get("beta_min", 0.05))
    swap_every = int(cfg.get("swap_interval", 50))
    burn_in    = int(cfg.burn_in)
    n_steps    = int(cfg.n_steps)
    plot_every = int(cfg.get("plot_every", 10000))
    seed       = int(cfg.seed)
    rng        = np.random.default_rng(seed)
    n_params   = int(cfg.get("n_corner_params", 10))
    xsec_only  = bool(cfg.get("xsec_only", False))

    betas = (list(cfg.temperatures) if cfg.get("temperatures", None)
             else _geom_temperatures(n_chains, beta_min))
    print(f"PT temperatures: {[f'{b:.3f}' for b in betas]}", flush=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    conns, workers = [], []
    for rank, beta in enumerate(betas):
        main_end, worker_end = mp.Pipe(duplex=True)
        w = mp.Process(target=_worker,
                       args=(rank, cfg_dict, beta, seed + rank, "", worker_end),
                       daemon=True)
        w.start()
        conns.append(main_end)
        workers.append(w)

    # Collect ready signals
    states, nlls_w = {}, {}
    mean = cov = par_names = bestfit_nll = L = None
    for conn in conns:
        msg = conn.recv()
        assert msg[0] == "ready"
        _, state, nll, pnames, bfnll, L_w, m = msg
        r = conns.index(conn)
        states[r]  = state.copy()
        nlls_w[r]  = nll
        if mean is None:
            mean        = m.copy()
            L           = L_w
            cov         = L @ L.T
            par_names   = pnames
            bestfit_nll = bfnll
    dim = len(mean)
    print(f"All {n_chains} workers ready  dim={dim}", flush=True)

    out_file  = _resolve_out(cfg, base_dir)
    # Use stem of out_file as subdirectory so concurrent seeds don't collide
    out_stem  = os.path.splitext(os.path.basename(out_file))[0]
    plots_dir = os.path.join(os.path.dirname(out_file), f"plots_pt_{out_stem}")
    os.makedirs(plots_dir, exist_ok=True)

    n_swaps_prop = np.zeros(n_chains - 1, dtype=int)
    n_swaps_acc  = np.zeros(n_chains - 1, dtype=int)

    def _send_step(phase: str, n: int):
        for r, conn in enumerate(conns):
            conn.send((f"step_{phase}", n, int(rng.integers(0, 2**31))))
        results = {}
        for r, conn in enumerate(conns):
            msg = conn.recv()
            assert msg[0] == "done"
            states[r]  = msg[1].copy()
            nlls_w[r]  = msg[2]
            results[r] = (msg[3],
                          msg[4] if len(msg) > 4 else None,
                          msg[5] if len(msg) > 5 else None)
        return results

    def _attempt_swaps():
        pairs = list(range(n_chains - 1))
        rng.shuffle(pairs)
        for i in pairs:
            j = i + 1
            log_acc = (betas[i] - betas[j]) * (nlls_w[i] - nlls_w[j])
            n_swaps_prop[i] += 1
            if np.log(rng.random() + 1e-300) < log_acc:
                n_swaps_acc[i] += 1
                states[i], states[j] = states[j].copy(), states[i].copy()
                nlls_w[i], nlls_w[j] = nlls_w[j], nlls_w[i]
                for idx in (i, j):
                    conns[idx].send(("swap", states[idx].copy(), nlls_w[idx]))
                for idx in (i, j):
                    ack = conns[idx].recv()
                    assert ack[0] == "swap_ack"

    # ── Burn-in ──────────────────────────────────────────────────────────────
    print(f"Burn-in: {burn_in} steps, swap every {swap_every}", flush=True)
    done_b = 0
    while done_b < burn_in:
        batch = min(swap_every, burn_in - done_b)
        r0    = _send_step("burn", batch)
        done_b += batch
        _attempt_swaps()
        acc0 = r0[0][0] / batch
        print(f"  burn-in {done_b}/{burn_in}  cold acc={acc0:.3f}"
              f"  swaps: {n_swaps_acc.sum()}/{n_swaps_prop.sum()}", flush=True)

    # ── Production ───────────────────────────────────────────────────────────
    print(f"Production: {n_steps} steps  plot every {plot_every}", flush=True)
    cold_chain = np.zeros((n_steps, dim))
    cold_nlls  = np.zeros(n_steps)
    n_acc_cold = 0
    done = 0
    next_plot  = plot_every

    while done < n_steps:
        batch  = min(swap_every, n_steps - done)
        r0     = _send_step("prod", batch)
        _attempt_swaps()

        c_chain, c_nlls = r0[0][1], r0[0][2]
        cold_chain[done:done+batch] = c_chain
        cold_nlls[done:done+batch]  = c_nlls
        n_acc_cold += r0[0][0]
        done       += batch

        print(f"  prod {done}/{n_steps}  cold acc={n_acc_cold/done:.3f}"
              f"  cold NLL={nlls_w[0]:.4f}"
              f"  swaps: {n_swaps_acc}/{n_swaps_prop}", flush=True)

        if done >= next_plot:
            plot_corner(
                cold_chain[:done], mean, cov, par_names, done,
                os.path.join(plots_dir, f"corner_{done:07d}.png"),
                n_params=n_params,
                xsec_only=xsec_only,
            )
            # Save checkpoint so partial chain is available before job ends
            chk_file = out_file.replace(".npz", f"_chk{done:07d}.npz")
            np.savez(
                chk_file,
                chain=cold_chain[:done], nll=cold_nlls[:done],
                par_names=np.array(par_names),
                bestfit_nll=float(bestfit_nll),
                temperatures=np.array(betas),
                acceptance_rate=float(n_acc_cold / done),
            )
            print(f"  checkpoint → {chk_file}", flush=True)
            next_plot += plot_every

    # Shutdown
    for conn in conns:
        conn.send(("quit",))
    for conn in conns:
        conn.recv()
    for w in workers:
        w.join(timeout=30)

    swap_rates = np.where(n_swaps_prop > 0,
                          n_swaps_acc / n_swaps_prop, 0.0)
    print(f"Swap acceptance rates: {swap_rates.tolist()}", flush=True)

    np.savez(
        out_file,
        chain=cold_chain, nll=cold_nlls,
        par_names=np.array(par_names),
        bestfit_nll=float(bestfit_nll),
        temperatures=np.array(betas),
        swap_acceptance_rates=swap_rates,
        acceptance_rate=float(n_acc_cold / n_steps),
    )
    print(f"Saved cold chain ({n_steps} steps, dim={dim}) → {out_file}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Independent chains coordinator
# ─────────────────────────────────────────────────────────────────────────────

def _run_independent(cfg: DictConfig, base_dir: str) -> None:
    n_chains = int(cfg.get("n_chains", 4))
    seed     = int(cfg.seed)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    out_file = _resolve_out(cfg, base_dir)
    tmp_dir  = os.path.join(os.path.dirname(out_file), "_chains")
    os.makedirs(tmp_dir, exist_ok=True)

    chain_paths = [os.path.join(tmp_dir, f"chain_rank{r:02d}.npz")
                   for r in range(n_chains)]

    workers = []
    for rank in range(n_chains):
        w = mp.Process(target=_worker,
                       args=(rank, cfg_dict, 1.0, seed + rank,
                             chain_paths[rank], None),
                       daemon=False)
        w.start()
        workers.append(w)

    for w in workers:
        w.join()

    print("Merging chains …", flush=True)
    all_chains, all_nlls = [], []
    par_names = bestfit_nll = None
    final_scales, acc_rates = [], []
    for p in chain_paths:
        d = np.load(p, allow_pickle=True)
        all_chains.append(d["chain"])
        all_nlls.append(d["nll"])
        if par_names is None:
            par_names   = d["par_names"]
            bestfit_nll = float(d["bestfit_nll"])
        final_scales.append(float(d["final_scale"]))
        acc_rates.append(float(d["acceptance_rate"]))

    np.savez(
        out_file,
        chain=np.concatenate(all_chains, axis=0),
        nll=np.concatenate(all_nlls, axis=0),
        par_names=par_names,
        bestfit_nll=bestfit_nll,
        n_chains=n_chains,
        per_chain_acceptance_rates=np.array(acc_rates),
        per_chain_final_scales=np.array(final_scales),
        acceptance_rate=float(np.mean(acc_rates)),
    )
    print(f"Saved merged chain ({n_chains} × {int(cfg.n_steps)} steps) → {out_file}",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="mcmc_pygundam", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        base_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    except Exception:
        base_dir = os.path.abspath(os.getcwd())

    mode = str(cfg.get("mode", "parallel_tempering"))
    print(f"MCMC mode: {mode}  n_chains={cfg.get('n_chains', 4)}", flush=True)

    mp.set_start_method("spawn", force=True)

    if mode == "parallel_tempering":
        _run_pt(cfg, base_dir)
    elif mode == "independent":
        _run_independent(cfg, base_dir)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'parallel_tempering' or 'independent'.")


if __name__ == "__main__":
    main()
