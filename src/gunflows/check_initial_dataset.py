import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from gunflows.likelihood_sampler import LikelihoodSampler
from gunflows.likelihood_sampler import pygundam_utils



parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, help='dataset file(s) to check (batch#.npz)', nargs='+')
parser.add_argument('-o', required=True, help='Name of the output folder for the plots.')

args = parser.parse_args()

out_dir = args.o
print("Using dataset file(s):", args.f)
# merge all files into one dictionary
all_data = {}
for file in args.f:
    data = np.load(file, allow_pickle=True)
    for key in data:
        if key in all_data:
            all_data[key] = np.concatenate((all_data[key], data[key]), axis=0)
        else:
            all_data[key] = data[key]

# Dictionary should look like:
            # "data": params_list,
            # "log_p": NLL_tot_list,
            # "log_q": baseline_NLL_list,
            # "cov": self.postfit_covariance_matrix,
            # "mean": self.postfit_parameter_values,
            # "par_names": self.get_parameter_names(),
            # "bestfit_nll": self.likelihood_at_bestfit

# plot histograms of each parameter with corelations to the next one only (just  a check)

if out_dir == "input":
    out_dir = os.path.join(os.environ.get("CONFIG_FOLDER"), "check_initial_dataset_plots")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
elif not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(f"Number of samples in the dataset: {all_data['data'].shape[0]}")
print(f"Output plots will be saved in: {out_dir}")

# Helper: robust 2D histogram with safe LogNorm and colorbar
def hist2d_logsafe(x, y, bins=60, weights=None, cmap="magma"):
    try:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
        pos = H > 0
        if not np.any(pos):
            # Fallback: linear norm, still show something
            im = plt.hist2d(x, y, bins=bins, weights=weights, cmap=cmap)[3]
            cb = plt.colorbar(im)
            cb.set_label("Counts (weighted)")
            return
        vmin = float(np.min(H[pos]))
        vmax = float(np.max(H))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmin >= vmax:
            # Fallback: linear norm
            im = plt.hist2d(x, y, bins=bins, weights=weights, cmap=cmap)[3]
        else:
            im = plt.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading='auto')
        cb = plt.colorbar(im)
        cb.set_label("Counts (weighted)")
    except Exception:
        # Last resort: draw without colorbar
        plt.hist2d(x, y, bins=bins, weights=weights, cmap=cmap)

num_params = all_data['data'].shape[1]
param_names = all_data['par_names']
# equalize medians of log_p and log_q
median_log_p = np.median(all_data['log_p'])
median_log_q = np.median(all_data['log_q'])
if np.isfinite(median_log_p) and np.isfinite(median_log_q):
    all_data['log_q'] += (median_log_p - median_log_q)

#plot log_p and log_q overlaid
plt.figure(figsize=(10, 6))
range_min = min(np.min(all_data['log_p']), np.min(all_data['log_q']))
range_max = max(np.max(all_data['log_p']), np.max(all_data['log_q']))
plt.hist(all_data['log_p'], bins=100, range=(range_min, range_max), density=True, alpha=0.5, label='log_p (NLL)', color='blue', histtype='stepfilled')
plt.hist(all_data['log_q'], bins=100, range=(range_min, range_max), density=True, alpha=0.5, label='log_q (baseline NLL)', color='orange', histtype='stepfilled')
# log scale y-axis
plt.yscale('log')
plt.xlabel('NLL')
plt.ylabel('Density')
plt.title('Distribution of log_p and log_q')
plt.legend()
plt.savefig(os.path.join(out_dir, '1_logp_logq_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

island_data = all_data['data'][(abs(all_data['log_p'] - all_data['log_q'])) > 250]
non_island_data = all_data['data'][(abs(all_data['log_p'] - all_data['log_q'])) <= 250]


weights = np.exp(all_data['log_p'] - all_data['log_q'])[(abs(all_data['log_p'] - all_data['log_q'])) <= 250]  # weights for the histograms
# Robustify weights: finite-only, clip extreme tail, and logweight < 250
weights = np.where(np.isfinite(weights), weights, 0.0)
if weights.ndim > 1:
    weights = weights.reshape(-1)
try:
    cap = np.quantile(weights, 0.995)
    if np.isfinite(cap) and cap > 0:
        weights = np.minimum(weights, cap)
except Exception:
    pass
# tw = float(np.sum(weights))
# if tw > 0:
#     weights = weights * (len(weights) / tw)

# tw = float(np.sum(weights
print(f"Number of island points detected (|log_p - log_q| > 250): {island_data.shape[0]}")
print(f"Number of points used for the main plots: {non_island_data.shape[0]}")
# rationalize names
param_names = [name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","").replace("-","_").replace(".","_") for name in param_names]
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    pass

bins_1d = 60
bins_2d = 60

for i in range(num_params):
    plt.figure(figsize=(12, 5))

    # Histogram of the parameter
    plt.subplot(1, 3, 1)
    xi = non_island_data[:, i]
    range_min = np.min(xi)
    range_max = np.max(xi)
    plt.hist(xi, bins=bins_1d, range=(range_min, range_max), weights=weights, density=True, histtype="step", color="#1f77b4")
    plt.title("|log_p-log_q| <= 250 (weighted and unweighted)")
    plt.xlabel(param_names[i])
    plt.ylabel('Density')
    plt.hist(xi, bins=bins_1d, range=(range_min, range_max), density=True, histtype="step", color='orange', alpha=0.5)

    # Scatter plot with the next parameter if it exists, otherwise with the previous one
    if i < num_params - 1:
        plt.subplot(1, 3, 2)
        x = non_island_data[:, i]
        y = non_island_data[:, i + 1]
        hist2d_logsafe(x, y, bins=bins_2d, weights=weights, cmap="magma")
        plt.xlabel(f'{param_names[i]}')
        plt.ylabel(f'{param_names[i+1]}')
    else:
        plt.subplot(1, 3, 2)
        x = non_island_data[:, i]
        y = non_island_data[:, i - 1]
        hist2d_logsafe(x, y, bins=bins_2d, weights=weights, cmap="magma")
        plt.xlabel(f'{param_names[i]}')
        plt.ylabel(f'{param_names[i-1]}')
    
    # third column with island points only
    plt.subplot(1, 3, 3)
    if island_data.shape[0] > 0:
        plt.hist(island_data[:, i], bins=bins_1d, range=(range_min, range_max), density=True, histtype="step", color='red')
        plt.title("|log_p-log_q| > 250 (unweighted)")
        plt.xlabel(param_names[i])
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No island points detected', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{param_names[i]}.png'), dpi=150, bbox_inches='tight')
    print(f"Saved plot for {param_names[i]}")
    plt.close()

