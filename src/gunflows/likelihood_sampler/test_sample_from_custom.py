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

# number of threads
if args.t:
    try:
        threads = int(args.t)
    except ValueError:
        raise ValueError(f"Invalid number of threads: {args.t}. Please provide an integer value.")
else:
    threads = 1

likelihood_sampler = LikelihoodSampler(config_file=args.c, override_files=args.of, data_is_asimov=data_is_asimov, threads=threads) # random seed not used in this mode (throwing outside gundam)
if likelihood_sampler.likelihood_interface is None:
    raise RuntimeError("Likelihood interface is not configured properly.")


# just for debugging
sample_names, samples = likelihood_sampler.get_list_of_samples()
print(f"Number of samples: {len(samples)}. Sample names:")
for name in sample_names:
    print(name)

bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
prior_parameter_values = likelihood_sampler.prior_parameter_values
postfit_covariance = likelihood_sampler.postfit_covariance_matrix
parameter_names = likelihood_sampler.get_parameter_names()

# compute log_det of the covariance matrix for normalization purpose
log_det_cov = np.linalg.slogdet(postfit_covariance)[1]  # log determinant
print(f"Debug. log_det_cov= {log_det_cov}")

start_time = time.time()
n = int(args.n)
N_n = 20000  # update plots every N_n samples
out_dir = 'img'
os.makedirs(out_dir, exist_ok=True)
# sample parameters from a custom distribution
NLL_list = []
params_list = []
logq_list = []
eigen_params_list = []
print(f"Throwing {n} samples from the covariance matrix...")
throws = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=n)
for i, throw in enumerate(throws):
    print(f"------------------------------Computing LH {i+1}/{n}...")
    # random_parameter_values = np.random.uniform(low=bestfit_parameter_values - 1.3*np.sqrt(np.diag(postfit_covariance)),
    #                                              high=bestfit_parameter_values + 1.3*np.sqrt(np.diag(postfit_covariance)),
    #                                              size=len(bestfit_parameter_values)
    #                                             )
    # random_parameter_values[:100] += 2.6 # Test: this should blow up the likelihood
    # get vector in the eigenspace of the covariance matrix
    log_q = pygundam_utils.log_multivariate_normal_pdf(throw, mean=bestfit_parameter_values, cov=postfit_covariance, with_log_det=True, precomputed_log_det=log_det_cov)
    # eigen_parameter_values = convert_to_eigenspace(random_parameter_values, mean=bestfit_parameter_values, cov=postfit_covariance)
    # print(f"Eigen parameter values: {pygundam_utils.big_vector_summary(eigen_parameter_values.tolist(),8)}")
    # print(f"Physi parameter values: {pygundam_utils.big_vector_summary(random_parameter_values.tolist(),8)}")
    # print(f"Bestf parameter values: {pygundam_utils.big_vector_summary(bestfit_parameter_values,8)}")
    NLL = likelihood_sampler.inject_params_and_compute_likelihood(throw, extend_continue=False)
    print(f"-log q: {log_q}")
    print(f"-log p: {NLL}", flush=True)
    # current_params = likelihood_sampler.get_current_parameter_values()
    # print(f"Current parameter values: {pygundam_utils.big_vector_summary(current_params,8)}")
    # for i in range (len(current_params)):
    #     if (current_params[i] != random_parameter_values[i]):
    #         print(f"Parameter {parameter_names[i]} mismatch. current: {current_params[i]} thrown: {random_parameter_values[i]} bf: {bestfit_parameter_values[i]}")
    #     if ( abs(current_params[i] - bestfit_parameter_values[i])<1e-10 ):
    #         print(f"Parameter {parameter_names[i]} is at best fit point. ")

    # nll_stat = likelihood_sampler.compute_stat_likelihood()
    # nll_syst = likelihood_sampler.compute_syst_likelihood()
    # print(f"Stat NLL: {nll_stat}, Syst NLL: {nll_syst}")
    # print(f"Total NLL: {nll_stat + nll_syst}")
    # # test: what's the nll at best fit point ?
    # nll_bestfit = likelihood_sampler.inject_params_and_compute_likelihood(bestfit_parameter_values)
    # print(f"Best fit NLL: {nll_bestfit}")
    # print(f"Best fit NLL: {likelihood_sampler.likelihood_at_bestfit}  [from lh sampler class]")
    # print(f"{likelihood_sampler.likelihood_interface.getSummary()}")


    # do append at the end of the loop
    # eigen_params_list.append(eigen_parameter_values)
    if not NLL == -1:
        params_list.append(throw)
        logq_list.append(log_q)
        NLL_list.append(NLL)

    # update plots every N_n samples
    if (i+1) % N_n == 0 or i == n-1:
        params_dict = likelihood_sampler.generate_dataset_dictionary(params_list, logq_list, NLL_list)
        output_file = args.o
        np.savez(output_file, **params_dict)
        print(f"Saved dataset to {output_file}")
        data = params_dict
        bestfit_nll = params_dict['bestfit_nll']
        parameter_names = data['par_names']
        params_array = np.array(data['data'])
        log_p = np.array(data['log_p'])
        log_q = np.array(data['log_q'])
        # draw NLL and gNLL
        draw_logp_logq(log_p, log_q, bestfit_nll, out_dir)




# print(f"bestfit_parameter_values: {pygundam_utils.big_vector_summary(bestfit_parameter_values)}")


end_time = time.time()
duration = end_time - start_time
print(f"Time for 1 LH evaluation: {duration/n*1000} ms")

# get the dictionary and save it
params_dict = likelihood_sampler.generate_dataset_dictionary(params_list, logq_list, NLL_list)
print(f"Saved dataset to {output_file}")

# plots

# NLL and gNLL
# draw all the parameter distributions, overlaying the prior and postfit values



# plot eigen parameters
# for i,param in enumerate(eigen_params_list):
#     plt.figure(figsize=(8, 6))
#     plt.hist(param, bins=50, density=True, color='lightblue', edgecolor='black')
#     plt.axvline(np.mean(param), color='red', linestyle='--', label='Mean')
#     plt.axvline(np.median(param), color='green', linestyle='--', label='Median')
#     plt.xlabel('Eigen Parameter Value')
#     plt.ylabel('Density')
#     plt.title(f"{parameter_names[i]}")
#     plt.legend()
#     os.makedirs(os.path.join(out_dir, 'eigens'), exist_ok=True)
#     plt.savefig(os.path.join(out_dir,'eigens', f'eigen_parameter_{i}.png'), dpi=100, bbox_inches='tight')








# plot the grid of parameters to show correlations
start_dim = 652
ndim = 8
fig, ax = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))
for i in range(start_dim,start_dim+ndim):
    for j in range(start_dim,start_dim+ndim):
        a = ax[i-start_dim, j-start_dim]
        if i == j:
            a.hist(params_array[:, i], bins=50, density=True, color='lightblue', edgecolor='black')
            a.axvline(bestfit_parameter_values[i], color='red', linestyle='--', label='Best Fit')
            a.axvline(prior_parameter_values[i], color='green', linestyle='--', label='Prior')
            a.set_xlabel(parameter_names[i])
            a.set_ylabel('Density')
            a.legend()
        else:
            a.hist2d(params_array[:, i], params_array[:, j], alpha=0.1, cmap='viridis', edgecolor='none')
            a.axvline(bestfit_parameter_values[i], color='red', linestyle='--', label='Best Fit')
            a.axhline(bestfit_parameter_values[j], color='red', linestyle='--')
            a.set_xlabel(parameter_names[i])
            a.set_ylabel(parameter_names[j])
            a.legend()
plt.tight_layout()
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'parameter_grid.png'), dpi=150)
plt.close(fig)