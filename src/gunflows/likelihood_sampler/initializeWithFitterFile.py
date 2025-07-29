import GUNDAM
import ROOT
import argparse
import time

from likelihoodSampler import LikelihoodSampler


#instantiate a likelihood sampler

parser = argparse.ArgumentParser()
parser.add_argument('-c', required=True, help='Config file path')
parser.add_argument('-of', nargs='+', help='Override config file paths')
args = parser.parse_args()

print("Using base config file:", args.c)
likelihood_sampler = LikelihoodSampler(args.c, args.of)

if likelihood_sampler.likelihood_interface is None:
    raise RuntimeError("Likelihood interface is not configured properly.")

sample_names, samples = likelihood_sampler.get_list_of_samples()
print(f"Number of samples: {len(samples)}. Sample names:")
for name in sample_names:
    print(name)


start_time = time.time()
likelihood_sampler.throw_n_from_covariance(100,printout=True)
end_time = time.time()

duration = end_time - start_time

print(f"{duration} s")
