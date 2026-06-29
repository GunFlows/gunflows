#!/usr/bin/env python3
"""
Standalone comparison: new MCMC chain vs NF samples vs Gaussian.
Produces a multi-page PDF (one figure per page):
  Page 1 — 1D marginals (last 10 params), matching make_paper_plots style
  Page 2a/b — Corner plots (5+5 params, viridis+LogNorm 2D), NF vs MCMC side by side
  Page 3 — ΔNLL distribution + log importance weights at MCMC points
  Page 4 — Correlation matrix comparison: NF / MCMC / difference (last 10 params)

No apptainer needed — pure numpy/matplotlib/torch.

Usage:
  python3 compare_mcmc_nf_marginals.py \
    --mcmc   <checkpoint.npz> \
    --nf     <samp_nf.npy> \
    --gauss  <samp_gaussian.npy> \
    --nlldata <nll_data.npz>   \   # optional; enables ΔNLL + weight plot
    --out    <output.pdf>
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

# Exact colours from make_paper_plots.py
COLORS = {"NF": "#1f77b4", "Gaussian": "#d62728", "MCMC": "#2ca02c"}
LBL_FS, TICK_FS, LEG_FS = 32, 26, 30


def _ax_fontsize(ax, label_fs=LBL_FS, tick_fs=TICK_FS, legend_fs=LEG_FS):
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(tick_fs)
    if ax.get_legend():
        for t in ax.get_legend().get_texts():
            t.set_fontsize(legend_fs)


def _short(s: str) -> str:
    raw = str(s).strip()
    for pre in ("Non-Linear Systematics/", "Linear Systematics/", "Systematics/", "Parameters/"):
        if raw.startswith(pre):
            raw = raw[len(pre):]
    if raw.lower().startswith("spline"):
        name = raw[len("Spline"):].strip().replace("_", " ").replace("mirror ", "").strip()
        return name
    return raw.lstrip("#0123456789_").strip()


# ── Page 1: 1D marginals (last 10 params) ────────────────────────────────────

def page_marginals(pdf, mcmc, samp_nf, samp_gauss, mean, std, par_names, sel, bins=60):
    n = len(sel)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows))
    axes = np.asarray(axes).flatten()

    for k, i in enumerate(sel):
        ax = axes[k]
        xnf = samp_nf[:, i] if i < samp_nf.shape[1] else np.array([])
        pools = [mcmc[:, i]]
        if len(xnf):
            pools.append(xnf)
        alld = np.concatenate(pools)
        lo = min(np.quantile(alld, 0.005), mean[i] - 4 * std[i])
        hi = max(np.quantile(alld, 0.995), mean[i] + 4 * std[i])
        rng = (float(lo), float(hi))

        if len(xnf):
            ax.hist(xnf, bins=bins, range=rng, density=True,
                    histtype="stepfilled", alpha=0.40, color=COLORS["NF"])
            ax.hist(xnf, bins=bins, range=rng, density=True,
                    histtype="step", linewidth=2.4, color=COLORS["NF"], label="NF")

        ax.hist(mcmc[:, i], bins=bins, range=rng, density=True,
                histtype="step", linewidth=2.4, color=COLORS["MCMC"], label="MCMC")

        xs = np.linspace(rng[0], rng[1], 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - mean[i]) / std[i]) ** 2)
                   / (std[i] * np.sqrt(2 * np.pi)),
                color=COLORS["Gaussian"], linewidth=2.4, linestyle="--", label="Gaussian")

        if samp_gauss is not None and i < samp_gauss.shape[1]:
            ax.hist(samp_gauss[:, i], bins=bins, range=rng, density=True,
                    histtype="step", linewidth=1.6, linestyle=":",
                    color=COLORS["Gaussian"])

        ax.set_xlabel(_short(par_names[i]))
        ax.set_ylabel("Density")
        ax.set_xlim(*rng); ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.tick_params(top=False, right=False)
        _ax_fontsize(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="upper right")
    _ax_fontsize(axes[0])
    fig.suptitle(f"MCMC ({len(mcmc):,} steps) vs NF vs Gaussian — last {n} parameters", fontsize=20)
    fig.tight_layout(pad=1.5)
    pdf.savefig(fig); plt.close(fig)


# ── Page 2: Corner plot (last 10 params), NF | MCMC side by side ─────────────

SIGMA = 4.5   # boundary width in units of postfit σ


def _corner_panel(ax_grid, samples, mean, std, sel, bins=40, title="", label_names=None):
    n   = len(sel)
    ss  = samples[:, sel]
    ms  = mean[sel]
    sts = std[sel]
    color = COLORS["NF"] if "NF" in title else COLORS["MCMC"]
    for row in range(n):
        for col in range(n):
            ax = ax_grid[row][col]
            if col > row:
                ax.axis("off"); continue
            lo1d, hi1d = ms[row] - SIGMA*sts[row], ms[row] + SIGMA*sts[row]
            if row == col:
                x = ss[:, row]; ok = np.isfinite(x)
                ax.hist(x[ok], bins=bins, range=(lo1d, hi1d), density=True,
                        histtype="stepfilled", alpha=0.45, color=color)
                ax.hist(x[ok], bins=bins, range=(lo1d, hi1d), density=True,
                        histtype="step", linewidth=1.2, color=color)
                xs = np.linspace(lo1d, hi1d, 200)
                ax.plot(xs, np.exp(-0.5*((xs-ms[row])/sts[row])**2)
                           / (sts[row]*np.sqrt(2*np.pi)),
                        color=COLORS["Gaussian"], linewidth=1.2, linestyle="--")
                ax.set_xlim(lo1d, hi1d)
            else:
                xd, yd = ss[:, col], ss[:, row]
                ok = np.isfinite(xd) & np.isfinite(yd)
                lo_x, hi_x = ms[col]-SIGMA*sts[col], ms[col]+SIGMA*sts[col]
                lo_y, hi_y = ms[row]-SIGMA*sts[row], ms[row]+SIGMA*sts[row]
                ax.hist2d(xd[ok], yd[ok], bins=bins,
                          range=[[lo_x, hi_x], [lo_y, hi_y]],
                          norm=LogNorm(), cmap="viridis")
                ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(labelbottom=(row == n-1), labelleft=(col == 0 and row > 0),
                           top=False, right=False, length=3, width=0.8, labelsize=8)
            if row == n-1 and label_names:
                ax.set_xlabel(label_names[col], fontsize=9)
            if col == 0 and row > 0 and label_names:
                ax.set_ylabel(label_names[row], fontsize=9)
    ax_grid[0][0].set_title(title, fontsize=12, color=color)


def _one_corner_page(pdf, mcmc, samp_nf, mean, std, par_names, sub_sel, tag):
    n    = len(sub_sel)
    labs = [_short(par_names[i]) for i in sub_sel]
    cell = 3.0
    fig, all_axes = plt.subplots(n, 2*n, figsize=(2*n*cell, n*cell))
    nf_axes   = [[all_axes[r, c]     for c in range(n)] for r in range(n)]
    mcmc_axes = [[all_axes[r, n+c]   for c in range(n)] for r in range(n)]
    _corner_panel(nf_axes,   samp_nf, mean, std, sub_sel, title="NF",   label_names=labs)
    _corner_panel(mcmc_axes, mcmc,    mean, std, sub_sel, title="MCMC", label_names=labs)
    fig.suptitle(f"Corner NF|MCMC — {tag}  (viridis+LogNorm, ±{SIGMA}σ)", fontsize=11)
    fig.tight_layout(pad=0.4)
    pdf.savefig(fig); plt.close(fig)


def page_corner(pdf, mcmc, samp_nf, mean, std, par_names, sel):
    # Split into two pages of 5 each
    half = len(sel) // 2
    _one_corner_page(pdf, mcmc, samp_nf, mean, std, par_names,
                     sel[:half], f"params {sel[0]}–{sel[half-1]}")
    _one_corner_page(pdf, mcmc, samp_nf, mean, std, par_names,
                     sel[half:], f"params {sel[half]}–{sel[-1]}")


# ── Page 3: ΔNLL + log-importance weights at MCMC points ─────────────────────

def page_nll_weights(pdf, mcmc_nll, bestfit_nll, nll_data, mcmc=None,
                     mean_phys=None, cov_phys=None, log_nf_at_mcmc=None):
    # bestfit_nll from MCMC checkpoint = 2*NLL (GUNDAM chi-squared); chain nlls = plain NLL.
    nll_best = float(nll_data["bestfit_nll"]) if nll_data is not None else bestfit_nll / 2
    d_nll_mcmc = mcmc_nll - nll_best
    d_nll_mcmc = d_nll_mcmc[np.isfinite(d_nll_mcmc) & (d_nll_mcmc >= 0) & (d_nll_mcmc < 1e6)]

    # log w(x) = log p(x) - log q(x) = -dNLL(x) - log q(x), centered to median=0
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    lw_nf = lw_g = None

    # NF weights at MCMC points (requires --nf_model)
    if log_nf_at_mcmc is not None and len(d_nll_mcmc) == len(log_nf_at_mcmc):
        raw = -d_nll_mcmc - log_nf_at_mcmc.astype(float)
        lw_nf = raw[np.isfinite(raw)]
        lw_nf -= np.median(lw_nf)

    # Gaussian weights at MCMC points (analytical)
    if mcmc is not None and len(d_nll_mcmc) and cov_phys is not None and mean_phys is not None:
        mcmc_trim = mcmc[:len(d_nll_mcmc)]
        n_dim = cov_phys.shape[0]
        diff  = (mcmc_trim - mean_phys).astype(np.float64)
        try:
            L_g = np.linalg.cholesky(cov_phys)
            y_g = np.linalg.solve(L_g, diff.T)
            log_det_g   = np.sum(np.log(np.diag(L_g)))
            log_q_gauss = -0.5*(np.sum(y_g**2, axis=0) + n_dim*np.log(2*np.pi)) - log_det_g
            raw_g = -d_nll_mcmc - log_q_gauss
            lw_g  = raw_g[np.isfinite(raw_g)]
            lw_g -= np.median(lw_g)
        except Exception as e:
            print(f"[weights] Gaussian log-q failed: {e}", flush=True)

    lo_w, hi_w = -10.0, 10.0

    if lw_nf is not None and len(lw_nf):
        ax.hist(lw_nf, bins=120, range=(lo_w, hi_w), density=True,
                histtype="stepfilled", alpha=0.35, color=COLORS["NF"])
        ax.hist(lw_nf, bins=120, range=(lo_w, hi_w), density=True,
                histtype="step", linewidth=2.4, color=COLORS["NF"],
                label="NF")
    if lw_g is not None and len(lw_g):
        ax.hist(lw_g, bins=120, range=(lo_w, hi_w), density=True,
                histtype="stepfilled", alpha=0.20, color=COLORS["Gaussian"])
        ax.hist(lw_g, bins=120, range=(lo_w, hi_w), density=True,
                histtype="step", linewidth=2.4, linestyle="--",
                color=COLORS["Gaussian"], label="Gaussian")
    ax.axvline(0, color="k", linewidth=1.0, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlim(lo_w, hi_w)
    ax.set_xlabel("log w", fontsize=16)
    ax.set_ylabel("Density")
    ax.legend(fontsize=LEG_FS, loc="upper left")
    _ax_fontsize(ax)
    fig.tight_layout(pad=1.5)
    pdf.savefig(fig); plt.close(fig)


# ── Page 4: Correlation matrices ─────────────────────────────────────────────

def page_correlations(pdf, mcmc, samp_nf, par_names, sel):
    labs = [_short(par_names[i]) for i in sel]
    corr_nf   = np.corrcoef(samp_nf[:, sel].T)
    corr_mcmc = np.corrcoef(mcmc[:, sel].T)
    diff      = corr_nf - corr_mcmc

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    n = len(sel)
    ticks = np.arange(n)

    for ax, mat, title, vmax in [
        (axes[0], corr_nf,   "NF correlations",   1.0),
        (axes[1], corr_mcmc, "MCMC correlations",  1.0),
        (axes[2], diff,      "NF − MCMC",          None),
    ]:
        cmap = "RdBu_r" if "−" in title else "viridis"
        vmax_ = vmax or np.abs(diff).max()
        vmin_ = -vmax_ if "−" in title else 0.0
        im = ax.imshow(mat, cmap=cmap, vmin=vmin_, vmax=vmax_,
                       interpolation="nearest", aspect="auto")
        ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=9)
        ax.set_title(title, fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Correlation matrices — last {n} parameters", fontsize=16)
    fig.tight_layout(pad=1.5)
    pdf.savefig(fig); plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc",    required=True)
    ap.add_argument("--nf",      required=True)
    ap.add_argument("--gauss",   default=None)
    ap.add_argument("--nlldata", default=None)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--bins",    type=int, default=60)
    ap.add_argument("--burnin_frac", type=float, default=0.1)
    ap.add_argument("--burnin",  type=int, default=None,
                    help="Fixed burn-in steps (overrides --burnin_frac)")
    ap.add_argument("--thin_to", type=int, default=None,
                    help="Thin chain to exactly this many samples after burn-in")
    ap.add_argument("--nf_model", default=None,
                    help="Path to NF checkpoint dir (enables NF log-prob at MCMC pts)")
    args = ap.parse_args()

    # Load MCMC chain
    data      = np.load(args.mcmc, allow_pickle=True)
    chain_all = data["chain"].astype(np.float64)
    par_names = [str(n) for n in data["par_names"]]
    mcmc_nll  = data["nll"].astype(np.float64)
    bestfit_nll = float(data.get("bestfit_nll", np.min(mcmc_nll)))
    n_steps   = len(chain_all)
    burn      = args.burnin if args.burnin is not None else int(args.burnin_frac * n_steps)
    mcmc      = chain_all[burn:]
    mcmc_nll  = mcmc_nll[burn:]
    if args.thin_to is not None and len(mcmc) > args.thin_to:
        step     = len(mcmc) // args.thin_to
        mcmc     = mcmc[::step][:args.thin_to]
        mcmc_nll = mcmc_nll[::step][:args.thin_to]
    print(f"MCMC: {len(mcmc)} steps after {burn} burn-in", flush=True)

    samp_nf    = np.load(args.nf).astype(np.float64)
    samp_gauss = np.load(args.gauss).astype(np.float64) if args.gauss else None
    print(f"NF samples: {samp_nf.shape}", flush=True)

    # Try default nlldata path if not provided
    nll_data = None
    nlldata_path = args.nlldata
    if nlldata_path is None:
        # try to find alongside the nf file
        candidate = Path(args.nf).parent / "nll_data.npz"
        if candidate.exists():
            nlldata_path = str(candidate)
    if nlldata_path:
        nll_data = np.load(nlldata_path, allow_pickle=True)
        print(f"NLL data loaded from {nlldata_path}", flush=True)

    nll_cache = Path(args.nf).parent / "nll_data.npz"
    if nll_cache.exists():
        _nd = np.load(nll_cache, allow_pickle=True)
        mean = _nd["bestfit"].astype(np.float64)
        cov  = _nd["cov"].astype(np.float64)
    else:
        mean = np.nanmean(mcmc, axis=0)
        cov  = np.cov(mcmc.T)
    std = np.sqrt(np.clip(np.diag(cov), 1e-14, None))

    # Last 10 parameters
    n_show = min(10, len(par_names))
    sel    = np.arange(len(par_names) - n_show, len(par_names))

    # Optional: evaluate NF log-prob at thinned MCMC points
    log_nf_at_mcmc = None
    if args.nf_model is not None:
        try:
            import sys, yaml
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import torch
            from gunflows.utils.build_flow import build_base, build_flow_layers, build_model

            cfg_path = Path(args.nf_model) / ".hydra" / "config.yaml"
            chk_path = Path(args.nf_model) / "checkpoints" / "best_model.pth"
            with open(cfg_path) as _f:
                _cfg = yaml.safe_load(_f)
            m = type("M", (), _cfg["experiment"]["model"])()

            _nd2    = np.load(nll_cache, allow_pickle=True) if nll_cache.exists() else nll_data
            nf_mean = _nd2["bestfit"].astype(np.float64)
            nf_cov  = _nd2["cov"].astype(np.float64)

            # Build a minimal dataset-like object matching _SamplingDataset
            phase_space_dim = list(range(int(m.total_dim) - len(list(range(100, 110))), int(m.total_dim)))
            # Use phase_space_dim from config if available (params 100-109)
            class _DS:
                pass
            ds = _DS()
            ds.phase_space_dim = list(range(100, 110))
            ds.list_dim_conditionnal = list(range(100))
            mean_t = torch.as_tensor(nf_mean, dtype=torch.float32)
            cov_t  = torch.as_tensor(nf_cov,  dtype=torch.float32)
            std_t  = torch.sqrt(torch.clamp(torch.diag(cov_t), min=1e-12))
            dinv   = torch.diag(1.0 / std_t)
            cov_std = dinv @ cov_t @ dinv
            ds.mean        = mean_t
            ds.std_per_dim = std_t
            ds.cholesky    = torch.linalg.cholesky(cov_std + 1e-6 * torch.eye(cov_std.shape[0]))

            n_spline = len(ds.phase_space_dim)
            tail  = torch.ones(n_spline) * float(m.tail_bound)
            base  = build_base(int(m.total_dim))
            flows = build_flow_layers(
                int(m.nflows), n_spline, int(m.hidden), int(m.nlayers), int(m.nbins),
                tail, n_context=int(m.total_dim) - n_spline,
            )
            kw = {k: int(getattr(m, k)) for k in ("n_context_flows", "hidden_dim", "n_hidden_layers") if hasattr(m, k)}
            device = torch.device("cpu")
            model  = build_model(base, flows, ds, bool(getattr(m, "context_transform", True)),
                                 False, **kw).to(device).eval()
            state    = torch.load(chk_path, map_location=device)
            log_norm = float(state.get("log_norm", torch.tensor(0.0)))
            model.load_state_dict(state)
            print(f"[nf_model] loaded {chk_path} (log_norm={log_norm:.4f})", flush=True)

            nf_std_np = std_t.numpy().astype(np.float64)
            x_eigen   = (mcmc - nf_mean) / nf_std_np
            x_sp  = torch.tensor(x_eigen[:, 100:110], dtype=torch.float32)
            x_ctx = torch.tensor(x_eigen[:, :100],    dtype=torch.float32)
            BS = 2000
            log_nf_at_mcmc = np.empty(len(mcmc), dtype=np.float64)
            with torch.no_grad():
                for i in range(0, len(mcmc), BS):
                    lq = model.log_prob(x_sp[i:i+BS], x_ctx[i:i+BS])
                    log_nf_at_mcmc[i:i+BS] = lq.cpu().numpy() + log_norm
            print(f"[nf_model] log_prob at {len(mcmc)} MCMC pts done", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[nf_model] failed: {e}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_marginals(pdf, mcmc, samp_nf, samp_gauss, mean, std, par_names, sel, bins=args.bins)
        page_corner(pdf, mcmc, samp_nf, mean, std, par_names, sel)
        page_nll_weights(pdf, mcmc_nll, bestfit_nll, nll_data,
                         mcmc=mcmc, mean_phys=mean, cov_phys=cov,
                         log_nf_at_mcmc=log_nf_at_mcmc)
        page_correlations(pdf, mcmc, samp_nf, par_names, sel)
    print(f"Saved {out} (5 pages)", flush=True)


if __name__ == "__main__":
    main()
