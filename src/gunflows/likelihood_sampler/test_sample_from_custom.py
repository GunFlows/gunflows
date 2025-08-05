import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from .likelihoodSampler import LikelihoodSampler


#instantiate a likelihood sampler

parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-n', required=True, help='Number of throws')
parser.add_argument('-o', required=True, help='Name of the output file')
parser.add_argument('-of', nargs='+', help='Override config file paths')
parser.add_argument('-t', help='Number of threads')
parser.add_argument('-a', action='store_true',help='Set data to prior, to be used for Asimov fits')
parser.add_argument('-s', type=int, help='Random seed')
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
n = 10
# sample parameters from a custom distribution
NLL_list = []
params_list = []
logq_list = []
for i in range(n):
    random_parameter_values = np.random.multivariate_normal(
        mean=bestfit_parameter_values,
        cov=postfit_covariance,
        size=1
    )[0]
    params_list.append(random_parameter_values)
    negative_log_q = 1 # dummy placeholder
    logq_list.append(negative_log_q)
    NLL = likelihood_sampler.inject_params_and_compute_likelihood(random_parameter_values)
    NLL_list.append(NLL)


end_time = time.time()
duration = end_time - start_time
print(f"Time for 1 LH evaluation: {duration/n*1000} ms")

# get the dictionary and save it
params_dict = likelihood_sampler.generate_dataset_dictionary(params_list, logq_list, NLL_list)
output_file = "test.npz"
# save the dictionary to npz file
np.savez(output_file, **params_dict)
print(f"Saved dataset to {output_file}")



