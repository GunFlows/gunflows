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
parser.add_argument('-b', help='Number of throws in a batch')
parser.add_argument('-of', nargs='+', help='Override config file paths')
parser.add_argument('-t', help='Number of threads')
parser.add_argument('-a', action='store_true',help='Set data to prior, to be used for Asimov fits')
args = parser.parse_args()


output_file = args.o
# Check if the output file has a .npz extension, if not, add it
if not output_file.endswith('.npz'):
    output_file += '.npz'
# remove .npz for the output directory
out_dir = os.path.splitext(output_file)[0] + '_plots'
print(f"Output dataset will be saved to: {output_file}")
print(f"Output plots will be saved to: {out_dir}")
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # raise RuntimeError(f"Output directory {out_dir} does not exist. Please create it first.")


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

bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
prior_parameter_values = likelihood_sampler.prior_parameter_values
postfit_covariance = likelihood_sampler.postfit_covariance_matrix
parameter_names = likelihood_sampler.get_parameter_names()

# compute log_det of the covariance matrix. Gonna use it later for normalization
log_det_cov = np.linalg.slogdet(postfit_covariance)[1]  # log determinant
print(f"Log determinant of the covariance matrix: {log_det_cov} (Used for normalization)")
if log_det_cov == -np.inf:
    raise ValueError("Log determinant of the covariance matrix is -inf. This indicates a singular matrix, which cannot be used for sampling.")
prefit_cov = likelihood_sampler.prefit_covariance_matrix
log_det = np.linalg.slogdet(prefit_cov)[1]  # log determinant of the prefit covariance matrix
print(f"Log determinant of the prefit covariance matrix: {log_det}")

n = int(args.n)
b = int(args.b) if args.b else n
if b > n:
    b = n



# save in a yaml file to the output folder: config file, overrides and working directory
# TODO: It's ok as long as there are few params, but this is a bit too manual...
same_folder_as_outputfile = os.path.dirname(os.path.abspath(output_file))
with open(os.path.join(same_folder_as_outputfile,"config_make_initial_dataset.yaml"), "w") as f:
    f.write("experiment:")
    f.write("  dataset:")
    f.write(f"    llh_config: {args.c}\n")
    f.write(f"    llh_overrides: {args.of}\n")
    f.write(f"    llh_cwd: {os.getcwd()}\n")
    f.write(f"    data_is_asimov: {data_is_asimov}\n")
    f.write(f"make_initial_dataset:\n")
    f.write(f"  total_throws: {n}\n")
    f.write(f"  batch_size: {b}\n")

print(f"Sampling and computing likelihoods for {n} throws in batches of {b}.")

# start loop of batches
throws = []
log_q_list = []
log_p_list = []
stat_NLL_list = []
syst_NLL_list = []

while len(throws) < n:
    i_global = len(throws)
    start_time = time.time()
    log_q_batch = []
    log_p_batch = []
    stat_NLL_list_batch = []
    syst_NLL_list_batch = []
    # Sample throws from the multivariate normal distribution
    b = min(b, n - len(throws))  # Ensure we don't exceed the total number of throws
    throws_batch = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=b)
    # Compute the log probabilities
    for i, throw in enumerate(throws_batch):
        i_global = len(throws) + i
        # print(f"first throw: {pygundam_utils.big_vector_summary(throw, 10)}")
        tot, stat_NLL, penalty_NLL = likelihood_sampler.inject_params_and_compute_likelihood(throw, extend_continue=False)
        NLL = penalty_NLL + stat_NLL
        while tot == -1:
            # I need to re-throw and replace the throw in the batch
            rethrow = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance)
            # print(f"re-throw: {pygundam_utils.big_vector_summary(rethrow, 10)}")
            tot, stat_NLL, penalty_NLL = likelihood_sampler.inject_params_and_compute_likelihood(rethrow, extend_continue=False)
            NLL = penalty_NLL + stat_NLL

            throw = rethrow
            throws_batch[i] = rethrow
        logq = pygundam_utils.log_multivariate_normal_pdf(throw, mean=bestfit_parameter_values, cov=postfit_covariance, with_log_det=True, precomputed_log_det=log_det_cov)
        log_q_batch.append(logq)
        log_p_batch.append(NLL)
        stat_NLL_list_batch.append(stat_NLL)
        syst_NLL_list_batch.append(penalty_NLL)
        print(f"Throw {i_global}: log_q = {logq}")
        print(f"                  log_p = {NLL}", flush=True)
        if (i_global >= 42 or i_global <= 72):
            print("42 is the answer to life, the universe and everything.")
            print(likelihood_sampler.likelihood_interface.getSummary())
    #test

    throws.extend(throws_batch)
    log_q_list.extend(log_q_batch)
    log_p_list.extend(log_p_batch)
    stat_NLL_list.extend(stat_NLL_list_batch)
    syst_NLL_list.extend(syst_NLL_list_batch)
    # At every batch, we also save the current state of the throws
    print(f"Processed {len(throws)}/{n} throws. Time per throw in last batch: {(time.time() - start_time)/b:.2f} seconds (batch size: {b})")
    dataset_dict = likelihood_sampler.generate_dataset_dictionary(throws, log_q_list, log_p_list)
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

    # draw syst and stat nll separately
    plt.figure(figsize=(10, 6))
    plt.hist(stat_NLL_list, alpha=0.7, density=True,
             color='lightblue', bins=100, edgecolor='black', label='Stat NLL Samples')
    plt.xlabel('Stat NLL')
    plt.ylabel('Density')
    plt.title(f'Stat NLL Distribution (throws: {len(stat_NLL_list)})')
    plt.legend()
    # mean and std
    mu, std = np.mean(stat_NLL_list), np.std(stat_NLL_list)
    plt.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')
    plt.axvline(mu + std, color='green', linestyle='--', label=f'Std: {std:.2f}')
    plt.axvline(mu - std, color='green', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'Stat_NLL_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.hist(syst_NLL_list, alpha=0.7, density=True,
             color='orange', bins=100, edgecolor='black', label='Syst NLL Samples')
    plt.xlabel('Syst NLL')
    plt.ylabel('Density')
    plt.title(f'Syst NLL Distribution (throws: {len(syst_NLL_list)})')
    plt.legend()
    # mean and std
    mu, std = np.mean(syst_NLL_list), np.std(syst_NLL_list)
    plt.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')
    plt.axvline(mu + std, color='green', linestyle='--', label=f'Std: {std:.2f}')
    plt.axvline(mu - std, color='green', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'Syst_NLL_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    # plot syst and stat vs. gNLL
    plt.figure(figsize=(10, 6))
    plt.scatter(stat_NLL_list, syst_NLL_list, alpha=0.5,
                color='blue', label='Stat vs Syst NLL')
    plt.xlabel('Stat NLL')
    plt.ylabel('Syst NLL')
    plt.title('Stat NLL vs Syst NLL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'Stat_vs_Syst_NLL.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.hist2d(stat_NLL_list, log_q, alpha=0.5,
                cmap='viridis', bins=100, label='Stat NLL vs gNLL', density=True)
    plt.xlabel('Stat NLL')
    plt.ylabel('gNLL')
    plt.title('Stat NLL vs gNLL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'Stat_NLL_vs_gNLL.png'), dpi=100, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    m1 = np.mean(syst_NLL_list)
    m2 = np.mean(log_q)
    plt.hist2d(syst_NLL_list, log_q - m2 + m1, alpha=0.5,
               cmap='viridis', bins=100, label='Syst NLL vs gNLL', density=True)
    plt.xlabel('Syst NLL')
    plt.ylabel('gNLL (shifted)')
    plt.title('Syst NLL vs gNLL')
    # overlay a diagonal line
    min_val = min(min(log_q), min(syst_NLL_list))
    max_val = max(max(log_q), max(syst_NLL_list))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
    plt.colorbar(label='Density')
    plt.tight_layout()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'Syst_NLL_vs_gNLL.png'), dpi=100, bbox_inches='tight')
    plt.close()








