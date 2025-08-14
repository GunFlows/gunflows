import GUNDAM
import ROOT
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from gunflows.likelihood_sampler import LikelihoodSampler
from gunflows.likelihood_sampler import pygundam_utils

# Start of the script
os.chdir(os.environ.get("CONFIG_FOLDER") )

# Instantiate a likelihood sampler
parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-n', required=True, help='Number of throws')
parser.add_argument('-o', required=True, help='Name of the output file')
parser.add_argument('-b', help='Number of throws ina batch')
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

# compute log_det of the covariance matrix. Gonna use it later for normalization
log_det_cov = np.linalg.slogdet(postfit_covariance)[1]  # log determinant
print(f"Log determinant of the covariance matrix: {log_det_cov} (Used for normalization)")

n = int(args.n)
b = int(args.b) if args.b else n
if b > n:
    b = n

out_dir = "img"

print(f"Sampling and computing likelihoods for {n} throws in batches of {b}.")

# start loop of batches
throws = []
log_q_list = []
log_p_list = []

while len(throws) < n:
    start_time = time.time()
    log_q_batch = []
    log_p_batch = []
    # Sample throws from the multivariate normal distribution
    b = min(b, n - len(throws))  # Ensure we don't exceed the total number of throws
    throws_batch = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=b)
    # Compute the log probabilities
    for i, throw in enumerate(throws_batch):
        NLL = likelihood_sampler.inject_params_and_compute_likelihood(throw, extend_continue=False)
        while NLL == -1:
            # I need to re-throw and replace the throw in the batch
            rethrow = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance)
            NLL = likelihood_sampler.inject_params_and_compute_likelihood(rethrow, extend_continue=False)
            throw = rethrow
            throws_batch[i] = rethrow
        logq = pygundam_utils.log_multivariate_normal_pdf(throw, mean=bestfit_parameter_values, cov=postfit_covariance, with_log_det=True, precomputed_log_det=log_det_cov)
        log_q_batch.append(logq)
        log_p_batch.append(NLL)
        print(f"log_q = {logq}")
        print(f"log_p = {NLL}", flush=True)
    #test

    throws.extend(throws_batch)
    log_q_list.extend(log_q_batch)
    log_p_list.extend(log_p_batch)
    # At every batch, we also save the current state of the throws
    print(f"Processed {len(throws)}/{n} throws. Time per throw in last batch: {(time.time() - start_time)/b:.2f} seconds (batch size: {b})")
    dataset_dict = likelihood_sampler.generate_dataset_dictionary(throws, log_q_list, log_p_list)
    output_file = args.o
    np.savez(output_file, **dataset_dict)
    print(f"Saved dataset to {output_file}")
    data = dataset_dict
    bestfit_nll = dataset_dict['bestfit_nll']
    parameter_names = data['par_names']
    params_array = np.array(data['data'])
    log_p = np.array(data['log_p'])
    log_q = np.array(data['log_q'])
    # draw NLL and gNLL
    pygundam_utils.draw_logp_logq(log_p, log_q, bestfit_nll, out_dir)




