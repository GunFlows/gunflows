to run the script that samples from NF, cov and MCMC, and then compares them with a certain configuration, do:

sbatch sample_and_compare.sh --config-name=config_name

example (for fds):
sbatch sample_and_compare.sh --config-name=sample_fds

the possibleconfig you can use are the names of the files in /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/configs without the .yaml extension
