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
args = parser.parse_args()

# Config file (Fitter output file)
print("Using base config file:", args.c)

# Output file
output_file = args.o

# Number of threads
if args.t:
    try:
        threads = int(args.t)
    except ValueError:
        raise ValueError(f"Invalid number of threads: {args.t}. Please provide an integer value.")
else:
    threads = 1

# input is an Asimov fit?
if args.a:
    data_is_asimov = True
else:
    data_is_asimov = False
likelihood_sampler = LikelihoodSampler(config_file=args.c, override_files=args.of, data_is_asimov=data_is_asimov, threads=threads)

# Number of throws
try:
    n = int(args.n)
except ValueError:
    raise ValueError(f"Invalid number of throws: {args.n}. Please provide an integer value.")

# Sanity check. Likelihood interface is configured properly?
if likelihood_sampler.likelihood_interface is None:
    raise RuntimeError("Likelihood interface is not configured properly.")

start_time = time.time()
params_list, weights_list, NLL_list = likelihood_sampler.throw_n_from_covariance(n,printout=False)
end_time = time.time()

duration = end_time - start_time

print(f"Total sampling time: {duration:.0f} seconds")
print(f"Time for 1 LH evaluation: {duration/n*1000:.0f} ms")

# get the dictionary and save it
params_dict = likelihood_sampler.generate_dataset_dictionary(params_list, weights_list, NLL_list)
# save the dictionary to npz file
np.savez(output_file, **params_dict)
print(f"Saved dataset to {output_file}")
