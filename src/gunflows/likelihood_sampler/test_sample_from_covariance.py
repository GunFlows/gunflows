import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from likelihoodSampler import LikelihoodSampler


#instantiate a likelihood sampler

parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-of', nargs='+', help='Override config file paths')
parser.add_argument('-a', action='store_true',help='Set data to prior')
args = parser.parse_args()

print("Using base config file:", args.c)
if args.a:
    data_is_asimov = True
else:
    data_is_asimov = False
likelihood_sampler = LikelihoodSampler(config_file=args.c, override_files=args.of, data_is_asimov=data_is_asimov)

if likelihood_sampler.likelihood_interface is None:
    raise RuntimeError("Likelihood interface is not configured properly.")

sample_names, samples = likelihood_sampler.get_list_of_samples()
print(f"Number of samples: {len(samples)}. Sample names:")
for name in sample_names:
    print(name)

bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
prior_parameter_values = likelihood_sampler.prior_parameter_values
postfit_covariance = likelihood_sampler.postfit_covariance_matrix

start_time = time.time()
n = 10000
params_list, weights_list, NLL_list = likelihood_sampler.throw_n_from_covariance(n,printout=False)
gNLL_list = [sum(weights) for weights in weights_list]
end_time = time.time()

duration = end_time - start_time

print(f"Time for 1 LH evaluation: {duration/n*1000} ms")

# get the dictionary and save it
params_dict = likelihood_sampler.generate_dataset_dictionary(params_list, weights_list, NLL_list)
output_file = "test.npz"
# save the dictionary to npz file
np.savez(output_file, **params_dict)
print(f"Saved dataset to {output_file}")













# draw all the parameter distributions, overlaying the prior and postfit values
os.makedirs('img', exist_ok=True)
parameter_names = likelihood_sampler.get_parameter_names()
params_array = np.array(params_list)
# draw NLL and gNLL
plt.figure(figsize=(8, 6))
plt.hist(NLL_list, alpha=0.7, density=True,
         color='lightblue', bins=100, edgecolor='black', label='NLL Samples')
plt.xlabel('NLL')
plt.ylabel('Density')
plt.title('NLL Distribution')
plt.savefig('img/NLL_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
plt.hist(gNLL_list, alpha=0.7, density=True,
         color='orange', bins=100, edgecolor='black', label='gNLL Samples')
plt.xlabel('NLL')
plt.ylabel('Density')
plt.title('gNLL Distribution')
plt.savefig('img/gNLL_distribution.png', dpi=100, bbox_inches='tight')
plt.figure(figsize=(8, 6))
# 2d histogram of NLL vs gNLL
plt.hist2d(NLL_list, gNLL_list, bins=100, cmap='viridis', density=True)
plt.colorbar(label='Density')
plt.xlabel('NLL')
plt.ylabel('gNLL')
# save pics
plt.savefig('img/NLL_gNLL_histogram.png', dpi=100, bbox_inches='tight')
plt.close()

for i, param_name in enumerate(parameter_names):
    if 10 < i < 690:
        continue
    print(f"Drawing parameter distribution for {param_name}...")
    plt.figure(figsize=(8, 6))
    # Draw histogram of parameter values
    plt.hist(params_array[:, i], alpha=0.7, density=True,
             color='lightblue', bins=100, edgecolor='black', label='Samples')

    # Overlay bestfit value as vertical line
    if i < len(bestfit_parameter_values):
        plt.axvline(bestfit_parameter_values[i], color='green', linestyle='-',
                    linewidth=2, label='Best Fit')
    # Overlay prior value as vertical line
    if i < len(prior_parameter_values):
        plt.axvline(prior_parameter_values[i], color='red', linestyle='--',
                    linewidth=2, label='Prior')

    plt.xlabel(param_name)
    plt.ylabel('N throws')
    plt.title(f'{param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(f'img/param_{i}_{param_name.replace("/", "_").replace(" ", "_")}.png',
                dpi=100, bbox_inches='tight')
    plt.close()

print(f"Generated {len(parameter_names)} parameter distribution plots in img/ directory")
