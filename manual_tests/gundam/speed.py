
import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt


from apps.gundam.likelihoodSampler import LikelihoodSampler
from apps.gundam import pygundam_utils

os.chdir(os.environ.get("CONFIG_FOLDER") )

# Instantiate a likelihood sampler
parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-n', required=True, help='Number of throws')
parser.add_argument('-o', required=True, help='Name of the output file')
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


bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
prior_parameter_values = likelihood_sampler.prior_parameter_values
postfit_covariance = likelihood_sampler.postfit_covariance_matrix
parameter_names = likelihood_sampler.get_parameter_names()

# compute log_det of the covariance matrix for normalization purpose
log_det_cov = np.linalg.slogdet(postfit_covariance)[1]  # log determinant

# dummy test: multivariate sampling from a 2x2 matrix
mean = [2.3,3.2]
cov = np.array([[0.1, 0.05], [0.05, 0.1]])
n_samples = 10
samples = np.random.multivariate_normal(mean, cov, n_samples)

n = int(args.n)
logp_list = []
logq_list = []
# Loop for generation of samples
start_time = time.time()
throws = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=n)
end_time = time.time()
duration_sampling = end_time - start_time

# Loop for computation of the weights
start_time = time.time()
for throw in throws:
    # throw = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance)
    logq = pygundam_utils.log_multivariate_normal_pdf(throw, mean=bestfit_parameter_values, cov=postfit_covariance)
    # logq_list.append(logq)
end_time = time.time()
duration_sampling_weight = end_time - start_time

# Loop for the computation of the likelihood
start_time = time.time()
for i, thrown_vector in enumerate(throws):
    NLL = likelihood_sampler.inject_params_and_compute_likelihood(thrown_vector, extend_continue=False)
    logp_list.append(NLL)
end_time = time.time()
duration_likelihood = end_time - start_time

print(f"Time for sampling {n} throws: {duration_sampling:.2f} seconds")
print(f"Time for computing {n}  weights: {duration_sampling_weight:.2f} seconds")
print(f"Time for likelihood evaluation of {n} throws: {duration_likelihood:.2f} seconds")
