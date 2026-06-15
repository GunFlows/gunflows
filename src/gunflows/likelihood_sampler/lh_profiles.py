import GUNDAM
import ROOT
import argparse
import time
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math

from gunflows.likelihood_sampler import LikelihoodSampler
from gunflows.likelihood_sampler import pygundam_utils


# --- uniform "paper" plotting style (mirrors make_paper_plots.py) ------------
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family":      "serif",
    "font.size":        16,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  12,
    "legend.frameon":   False,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "axes.linewidth":   1.4,
    "axes.labelpad":    10,
    "lines.linewidth":  2.0,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})


def _clean_name(name):
    """Drop '_' and '#' from a systematic-parameter name for labels/legends."""
    return re.sub(r"\s+", " ", str(name).replace("#", " ").replace("_", " ")).strip()

# Start of the script
# save current directory
cwd = os.getcwd()
os.chdir(os.environ.get("CONFIG_FOLDER") )

# Instantiate a likelihood sampler
parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-o', required=True, help='Name of the output folder')
parser.add_argument('-n', help='Number of samples per profile', default=10)
parser.add_argument('-s', help='width of profile in sigmas', default=3)
parser.add_argument('-of', nargs='+', help='Override config file paths')
parser.add_argument('-t', help='Number of threads')
parser.add_argument('-a', action='store_true',help='Set data to prior, to be used for Asimov fits')
parser.add_argument('-x', action='store_true',help='Only xsec parameters')
args = parser.parse_args()


print("Using base config file:", args.c)
# Asimov or not?
if args.a:
    data_is_asimov = True
else:
    data_is_asimov = False

# number of threads
if args.t:
    try:
        threads = int(args.t)
    except ValueError:
        raise ValueError(f"Invalid number of threads: {args.t}. Please provide an integer value.")
else:
    threads = 1

# Finally initialize the likelihood sampler
likelihood_sampler = LikelihoodSampler(config_file=args.c, override_files=args.of, data_is_asimov=data_is_asimov, threads=threads) # random seed not used in this mode (throwing outside gundam)
if likelihood_sampler.likelihood_interface is None:
    raise RuntimeError("Likelihood interface is not configured properly.")

print("GUNDAM CONFIG:")
print(likelihood_sampler.get_gundam_config_yaml())

n = int(args.n)

if args.s:
    try:
        w = float(args.s)
    except ValueError:
        raise ValueError(f"Invalid value for sigma: {args.s}. Please provide a float value.")
else:
    w = 3.0

print(f"\n\nMaking NLL profile plots with {n} points in a range of +-{w} sigmas around the best fit point.\n\n")

# go back to original directory
os.chdir(cwd)
output_folder = args.o
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.makedirs(output_folder+"/flux", exist_ok=True)
os.makedirs(output_folder+"/det", exist_ok=True)
os.makedirs(output_folder+"/xsec", exist_ok=True)

bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
prior_parameter_values = likelihood_sampler.prior_parameter_values
postfit_covariance = likelihood_sampler.postfit_covariance_matrix
parameter_names = likelihood_sampler.get_parameter_names()

bf = bestfit_parameter_values

# compute likelihood at best fit
likelihood_at_bestfit,_,_ = likelihood_sampler.inject_params_and_compute_likelihood(bf,extend_continue=False)
print(f"nll bf: {likelihood_at_bestfit:.2f} (should be 0 for Asimov fit)")

# initialize parameter values to bf
parameter_values = bf.copy()

dummy = [1] * len(bf)  # dummy list of 1s


for i, parameter_name in enumerate(parameter_names):
    if args.x and "Cross-Section" not in parameter_name:
        continue    
    print(f"Computing profile for parameter {i+1}/{len(parameter_names)}: {parameter_name}",end=' ')
    phys_range_min, phys_range_max = likelihood_sampler.get_parameter_physical_range(parameter_name)
    # print(f"- phys range: [{phys_range_min:.2f}, {phys_range_max:.2f}]", end=' ')
    nll_list = []
    points = []
    parameter_range = [bf[i] - w*math.sqrt(postfit_covariance[i][i]), bf[i] + w*math.sqrt(postfit_covariance[i][i])]
    print(f"- scan: [{parameter_range[0]:.2f} , {parameter_range[1]:.2f}]", end=' ')
    step = (parameter_range[1] - parameter_range[0]) / n
    parameter_values = bf.copy()
    for j in range(n+1):
        point = j*step + parameter_range[0]
        # replace the i-th parameter value with the point
        parameter_values[i] = point
        nll, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(parameter_values,extend_continue=False)
        if nll != -1:
            nll_list.append(nll)
            points.append(point)
    # plot and save the profile
    stripped_parameter_name = parameter_name.split('#')[-1]
    clean_name = _clean_name(parameter_name)
    plt.figure()
    plt.plot(points, nll_list, label=clean_name, marker='.')
    plt.axhline(y=likelihood_at_bestfit, color='r', linestyle='--', label='Best fit likelihood')
    plt.axvline(x=bf[i] - 1*math.sqrt(postfit_covariance[i][i]), color='g', linestyle='-', label=r"$\pm 1 \sigma$")
    plt.axvline(x=bf[i] + 1*math.sqrt(postfit_covariance[i][i]), color='g', linestyle='-')
    plt.axvline(x=bf[i] - 2*math.sqrt(postfit_covariance[i][i]), color=('green',0.5), linestyle='-', label=r"$\pm 2 \sigma$")
    plt.axvline(x=bf[i] + 2*math.sqrt(postfit_covariance[i][i]), color=('green',0.5), linestyle='-')
    plt.axvline(x=prior_parameter_values[i], color=('orange',0.5), linestyle='--', label='Prior')
    plt.axvline(x=bf[i], color='orange', linestyle='--', label='Best fit')
    if phys_range_min > parameter_range[0]:
        plt.axvline(x=phys_range_min, color='black', linestyle=':', label='Physical limit')
        # shade the unphysical region
        plt.fill_betweenx([min(nll_list), max(nll_list)], parameter_range[0], phys_range_min, color='gray', alpha=0.3)
    if phys_range_max < parameter_range[1]:
        plt.axvline(x=phys_range_max, color='black', linestyle=':', label='Physical limit')
        # shade the unphysical region
        plt.fill_betweenx([min(nll_list), max(nll_list)], phys_range_max, parameter_range[1], color='gray', alpha=0.3)
    plt.xlabel(clean_name)
    plt.ylabel(r'$-\log(\mathcal{L}_p)$')
    plt.legend()
    if "Flux" in parameter_name:
        plt.savefig(os.path.join(output_folder, "flux", f'{stripped_parameter_name}.png'))
        print(f" -> {output_folder}/flux/{stripped_parameter_name}.png")
    elif "Det" in parameter_name:
        plt.savefig(os.path.join(output_folder, "det", f'{stripped_parameter_name}.png'))
        print(f" -> {output_folder}/det/{stripped_parameter_name}.png")
    elif "Cross-Section" in parameter_name:
        plt.savefig(os.path.join(output_folder, "xsec", f'{stripped_parameter_name}.png'))
        print(f" -> {output_folder}/xsec/{stripped_parameter_name}.png")
    else:
        plt.savefig(os.path.join(output_folder, f'{stripped_parameter_name}.png'))
        print(f" -> {output_folder}/{stripped_parameter_name}.png")
    plt.close()
