import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from gunflows.likelihood_sampler import LikelihoodSampler
from gunflows.likelihood_sampler import pygundam_utils

# Start of the script
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

n = int(args.n)

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
bf[0] = bf[0] + 1e-15
likelihood_at_bestfit,_,_ = likelihood_sampler.inject_params_and_compute_likelihood(bf,extend_continue=False)
print(f"nll bf: {likelihood_at_bestfit:.2f} (should be 0 for Asimov fit)")
bf[0] = bf[0] + 4e-14
likelihood_at_bestfit,_,_ = likelihood_sampler.inject_params_and_compute_likelihood(bf,extend_continue=False)
print(f"nll bf: {likelihood_at_bestfit:.2f} (should be 0 for Asimov fit)")

# initialize parameter values to bf
parameter_values = bf.copy()

dummy = [1] * len(bf)  # dummy list of 1s


for i, parameter_name in enumerate(parameter_names):
    print(f"Computing profile for parameter {i+1}/{len(parameter_names)}: {parameter_name}",end=' ')
    nll_list = []
    points = []
    parameter_range = [bf[i] - 2*math.sqrt(postfit_covariance[i][i]), bf[i] + 3*math.sqrt(postfit_covariance[i][i])]
    print(f": [{parameter_range[0]:.2f} , {parameter_range[1]:.2f}]", end=' ')
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
    plt.figure()
    plt.plot(points, nll_list, label=f"Profile {stripped_parameter_name}", marker='.')
    plt.axhline(y=likelihood_at_bestfit, color='r', linestyle='--', label='Best fit likelihood')
    plt.axvline(x=bf[i] - 1*math.sqrt(postfit_covariance[i][i]), color='g', linestyle='-', label=r"$\pm 1 \sigma$")
    plt.axvline(x=bf[i] + 1*math.sqrt(postfit_covariance[i][i]), color='g', linestyle='-')
    plt.axvline(x=bf[i] - 2*math.sqrt(postfit_covariance[i][i]), color=('green',0.5), linestyle='-', label=r"$\pm 2 \sigma$")
    plt.axvline(x=bf[i] + 2*math.sqrt(postfit_covariance[i][i]), color=('green',0.5), linestyle='-')
    plt.axvline(x=prior_parameter_values[i], color=('orange',0.5), linestyle='--', label='Prior')
    plt.axvline(x=bf[i], color='orange', linestyle='--', label='Best fit')
    plt.xlabel(parameter_name)
    plt.ylabel('nll')
    plt.title(f'Profile of {parameter_name}')
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
    plt.close()
