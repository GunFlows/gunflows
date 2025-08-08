

try:
    import GUNDAM
except ImportError:
    raise ImportError("GUNDAM module not found. Please ensure GUNDAM is properly installed and in your Python path.")
try:
    import ROOT
except ImportError:
    raise ImportError("ROOT module not found. Please ensure ROOT/PyROOT is properly installed and in your Python path.")
import argparse
from .pygundam_utils import *
from tqdm import tqdm
import sys
import time


class LikelihoodSampler:
    def __init__(self, config_file, override_files=None, threads=1, data_is_asimov=False, seed=None):
        self.likelihood_interface = None
        self.cb = None
        self.cr = None
        self.fitter = None
        self.propagator = None
        self.fitter_root_file = None
        self.data_is_asimov = data_is_asimov  # Set to True if using asimov
        self.config_file = config_file
        self.override_files = override_files if override_files else []
        self.prior_parameter_values = None
        self.postfit_parameter_values = None
        self.postfit_covariance_matrix = None
        self.likelihood_at_bestfit = None

        GUNDAM.setNumberOfThreads(threads)
        GUNDAM.setLightOutputMode(True)
        
        self.app = GUNDAM.GundamApp("GUNDAM: likelihood sampler")

        # read config from config file (.yaml) or Fitter output file (.root)
        if config_file.endswith('.yaml'):
            self.configure_using_yaml()
        elif config_file.endswith('.root'):
            self.configure_using_root()
        else:
            raise ValueError("Unsupported config file format. Use .yaml or .root")

        # Set the seed for reproducibility
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                raise ValueError(f"Invalid seed value: {seed}. Please provide an integer value.")
        else:
            seed = int(time.time() * 1000)  # Use current time in milliseconds
        ROOT.gRandom.SetSeed(seed)
        print(f"Random seed set to: {seed}")

        # If the input is a config file, with no Fitter output ROOT file, data HAS to be Asimov
        if not self.data_is_asimov and self.fitter_root_file is None:
            raise ValueError("Data is not set to Asimov, but no root file provided. Please provide a root file with data histograms.")

        if self.fitter is None:
            raise RuntimeError("Fitter engine is not configured properly.")

        # Do I need this?
        # fitter.getLikelihoodInterface().getModelPropagator().setEnableEigenToOrigInPropagate( false );

        # Initialize the fitter engine and get the likelihood interface
        self.fitter.initialize()
        self.likelihood_interface = self.fitter.getLikelihoodInterface()

        # Get the model propagator from the likelihood interface
        self.propagator = self.likelihood_interface.getModelPropagator()

        # The following needs the propagator to be initialized
        self.load_data_histograms(self.data_is_asimov)

        # Load the postfit covariance matrix into the propagator
        self.load_postfit_covariance_in_propagator()

        print(f"Number of parameters in the likelihood interface: {self.get_number_of_parameters()}")
        print("Propagator covariance matrix shape: ", self.propagator.getParametersManager().getGlobalCovarianceMatrix().GetNrows(), "x", self.propagator.getParametersManager().getGlobalCovarianceMatrix().GetNcols())

        # Open an output file (will be added later)
        # self.app.openOutputFile(self.output_file)
        # self.app.writeAppInfo()

        ######################################
        #        INITIALIZATION DONE         #
        ######################################

        # Print out parameters at prior
        self.prior_parameter_values = self.get_current_parameter_values()
        print(f"Parameters at prior:{big_vector_summary(self.prior_parameter_values)}")
        print(f"Current parameter values: {big_vector_summary(self.get_current_parameter_values())}")
        NLL_syst = self.compute_syst_likelihood()
        NLL_stat = self.compute_stat_likelihood()
        print(f"NLL: {NLL_stat} (stat) + {NLL_syst} (syst) = {NLL_stat + NLL_syst}")
        # Print out parameters at best fit point
        self._load_bestfit_parameter_values_()
        print(f"Parameters at best fit:{big_vector_summary(self.postfit_parameter_values)}")
        print(f"Current parameter values: {big_vector_summary(self.get_current_parameter_values())}")
        NLL_syst = self.compute_syst_likelihood()
        NLL_stat = self.compute_stat_likelihood()
        print(f"NLL: {NLL_stat} (stat) + {NLL_syst} (syst) = {NLL_stat + NLL_syst}")
        self.likelihood_at_bestfit = NLL_stat + NLL_syst

        # Reset the prior values to the postfit values
        self.reset_prior_values(self.postfit_parameter_values)
        print("WARNING: Prior values reset to postfit values!")
        print("LikelihoodSampler initialized successfully.")


    def configure_using_yaml(self):
        print("Using base config file:", self.config_file)

        self.cb = GUNDAM.ConfigUtils.ConfigBuilder(self.config_file)
        if (self.override_files is not None):
            for override_file in self.override_files:
                print("Using override:", override_file)
                self.cb.override(override_file)

        # Config reader setup:
        self.cr = GUNDAM.ConfigUtils.ConfigReader(self.cb.getConfig())
        self.cr.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
        fitter_engine_config = self.cr.fetchValueConfigReader("fitterEngineConfig")
        # Fitter setup:
        self.fitter = GUNDAM.FitterEngine()
        self.fitter.setConfig(fitter_engine_config)
        self.fitter.configure()

    def reset_prior_values(self, values):
        """
        Reset the prior parameter values to the given values.
        This is useful if you want to change the prior values after initialization.
        """
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        n = 0
        previous_priors = []
        for par_set in self.propagator.getParametersManager().getParameterSetsList():
            if par_set.isEnabled():
                for par in par_set.getParameterList():
                    if par.isEnabled():
                        previous_priors.append(par.getPriorValue())
                        par.setPriorValue(values[n])
                        n += 1
        if n != len(values):
            # If the number of values does not match, reset to previous priors
            self.reset_prior_values(self.prior_parameter_values)
            raise ValueError(f"reset_prior_values: Number of values provided ({len(values)}) does not match the number of parameters ({n}).")

    def inject_parameter_values(self, values):
        """
        Inject the given vector of parameter values into the propagator.
        """
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        n = 0
        current = self.get_current_parameter_values()
        for par_set in self.propagator.getParametersManager().getParameterSetsList():
            if par_set.isEnabled():
                for par in par_set.getParameterList():
                    if par.isEnabled():
                        par.setParameterValue(values[n])
                        n += 1
        if n != len(values):
            # If the number of values does not match, reset to previous values
            self.inject_parameter_values(current)
            raise ValueError(f"inject_parameter_values: Number of values provided ({len(values)}) does not match the number of parameters ({n}).")


    def configure_using_root(self):
        print("Extracting config from root file:", self.config_file)
        self.fitter_root_file = ROOT.TFile(self.config_file, "READ")
        if not self.fitter_root_file.IsOpen():
            raise FileNotFoundError(f"Could not open root file: {self.config_file}")

        config_tnamed = self.fitter_root_file.Get("gundamFitter/unfoldedConfig_TNamed")
        if not config_tnamed:
            config_tnamed = self.fitter_root_file.Get("gundam/config_TNamed")
        if not config_tnamed:
            config_tnamed = self.fitter_root_file.Get("gundamFitter/unfoldedConfig_TNamed")
        if not config_tnamed:
            config_tnamed = self.fitter_root_file.Get("gundam/config/unfoldedJson_TNamed")
        if not config_tnamed:
            raise RuntimeError("Could not find config TNamed in the root file.")
        else:
            print("Found config TNamed in ROOT file")

        # Read the configuration from the output file
        config_json = GUNDAM.GenericToolbox.Json.readConfigJsonStr(config_tnamed.GetTitle())
        if not config_json:
            raise RuntimeError("Failed to read config JSON from TNamed title.")
        self.cb = GUNDAM.ConfigUtils.ConfigBuilder(config_json)
        for override_file in self.override_files:
            print("Using override:", override_file)
            self.cb.override(override_file)
        self.cr = GUNDAM.ConfigUtils.ConfigReader(self.cb.getConfig())
        self.cr.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
        fitter_engine_config = self.cr.fetchValueConfigReader("fitterEngineConfig")
        # Fitter setup:
        self.fitter = GUNDAM.FitterEngine()
        self.fitter.setConfig(fitter_engine_config)
        self.fitter.configure()

    def load_postfit_covariance_in_propagator(self):
        if self.fitter_root_file is None:
            raise ValueError("Postfit covariance matrix can only be loaded from a root file.")
        postfit_covariance_matrix = self.fitter_root_file.Get("FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D")
        if not postfit_covariance_matrix:
            raise RuntimeError("Postfit covariance matrix not found in the root file [searched in \"FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D\"].")
        tmatrix = convert_TH2D_to_TMatrix(postfit_covariance_matrix)
        self.propagator.getParametersManager().setGlobalCovarianceMatrix(tmatrix)
        # convert the covariance matrix to a list of lists
        postfit_covariance_matrix = []
        n_rows = tmatrix.GetNrows()
        n_cols = tmatrix.GetNcols()
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                row.append(tmatrix[i,j])
            postfit_covariance_matrix.append(row)
        self.postfit_covariance_matrix = postfit_covariance_matrix
        print("Post-Fit covariance matrix loaded into the propagator.")

    def get_number_of_parameters(self):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        n = 0
        for par_set in self.propagator.getParametersManager().getParameterSetsList():
            if not par_set.isEnabled():
                continue
            for par in par_set.getParameterList():
                if not par.isEnabled():
                    continue
                n += 1
        return n

    def get_parameter_names(self):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        parameter_names = []
        for par_set in self.propagator.getParametersManager().getParameterSetsList():
            if not par_set.isEnabled():
                continue
            for par in par_set.getParameterList():
                if not par.isEnabled():
                    continue
                parameter_names.append(par.getFullTitle())
        return parameter_names

    def _load_bestfit_parameter_values_(self):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        if self.fitter_root_file is None:
            print("WARNING: No root file provided. Returning prior as best fit parameter values.")
        self.postfit_parameter_values = self.prior_parameter_values
        par_list_tnamed = self.fitter_root_file.Get("FitterEngine/postFit/parState_TNamed")
        if not par_list_tnamed:
            raise RuntimeError("Post-fit parameter values not found in the root file [searched in \"FitterEngine/postFit/parState_TNamed\"].")
        par_list_json = GUNDAM.GenericToolbox.Json.readConfigJsonStr(par_list_tnamed.GetTitle())
        self.propagator.getParametersManager().injectParameterValues(par_list_json, quietVerbose_=True)
        print("WARNING: Post-Fit parameter values injected as current parameter values!")
        # Now the current parameter values should be updated to the best fit values
        self.postfit_parameter_values = self.get_current_parameter_values()


    def get_current_parameter_values(self):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        values = []
        for par_set in self.propagator.getParametersManager().getParameterSetsList():
            if not par_set.isEnabled():
                continue
            for par in par_set.getParameterList():
                if not par.isEnabled():
                    continue
                values.append(par.getParameterValue())
        return values

    def get_list_of_samples(self):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        sample_names = []
        samples = []
        for sample in self.propagator.getSampleSet().getSampleList():
            if not sample.isEnabled():
                continue
            sample_names.append(sample.getName())
            samples.append(sample)
        return sample_names, samples

    def load_data_histograms(self, data_is_asimov):
        # Set the data as asimov (prior)
        self.fitter.getLikelihoodInterface().setForceAsimovData(True)
        if data_is_asimov:
            print("Data is set to Asimov priors.")
            return
        if self.fitter_root_file is None:
            print("WARNING: No root file provided. Data is set to Asimov priors.")
            return
        # Load data histograms from the root file
        # Loop through the samples
        sample_names, samples = self.get_list_of_samples()
        for sample_name, sample in zip(sample_names, samples):
            # Skip if the sample is not enabled
            if not sample.isEnabled():
                continue
            # Get the data histogram for the sample
            data_histogram = self.fitter_root_file.Get(f"FitterEngine/preFit/data/{sample_name}/histogram")
            if not data_histogram:
                raise RuntimeError(f"Data histogram for sample '{sample_name}' not found in the root file.")
            # sanity check: the data histogram must have the same binning as the model histogram
            n_bins_data = data_histogram.GetNbinsX() # ROOT function of TH1D class
            n_bins_model = sample.getHistogram().getNbBins() # GUNDAM function of Histogram class
            if n_bins_data != n_bins_model:
                raise RuntimeError(f"Data histogram for sample '{sample_name}' has {n_bins_data} bins, but model histogram has {n_bins_model} bins.\nPossible mismatch in fitter and LH sampelr configs!")
            bin_content_list = sample.getHistogram().getBinContentList()
            # loop and replace contents
            print(f"DEBUG| sample {sample_name}")
            for i in range(n_bins_data):
                bin_content = data_histogram.GetBinContent(i+1)
                bin_error = data_histogram.GetBinError(i+1)
                current_bin_content = bin_content_list[i].sumWeights
                current_bin_error = bin_content_list[i].sqrtSumSqWeights
                print(f"DEBUG| bin {i}: {current_bin_content:.2f} -> {bin_content:.2f} | {current_bin_error:.2f} -> {bin_error:.2f}")
                bin_content_list[i].sumWeights = bin_content  # this replaces the bin content in the sample histogram
                bin_content_list[i].sqrtSumSqWeights = bin_error  # I THINK this should be the bin error...

    def throw_one_from_covariance(self, printout=False):
        # the following throws parameters from the covariance matrix
        weights = self.propagator.getParametersManager().throwParametersFromGlobalCovariance()
        # the following propagates the parameters and computes the likelihood
        self.likelihood_interface.propagateAndEvalLikelihood()
        NLL_stat = self.fitter.getLikelihoodInterface().getBuffer().statLikelihood
        NLL_syst = self.fitter.getLikelihoodInterface().getBuffer().penaltyLikelihood
        parameter_values = self.get_current_parameter_values()
        NLL_tot = NLL_stat + NLL_syst

        if printout:
            print(f"Statistical NLL: {NLL_stat}, Systematic NLL: {NLL_syst}")
            print(f"--Params : {big_vector_summary(parameter_values)}")
            print(f"--Weights: {big_vector_summary(weights)}")

        return parameter_values, weights, NLL_tot

    def compute_stat_likelihood(self):
        self.likelihood_interface.propagateAndEvalLikelihood()
        return self.fitter.getLikelihoodInterface().getBuffer().statLikelihood
    def compute_syst_likelihood(self):
        self.likelihood_interface.propagateAndEvalLikelihood()
        return self.fitter.getLikelihoodInterface().getBuffer().penaltyLikelihood


    def throw_n_from_covariance(self, n, printout=False):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        if self.likelihood_interface is None:
            raise RuntimeError("The likelihood interface is not initialized.")
        # Set custom thrower!
        self.propagator.getParametersManager().setThrowerAsCustom()
        # Add a simple progress bar


        params_list = []
        weights_list = []
        NLL_tot_list = []
        disable_tqdm = False if sys.stdout.isatty() else True  # Use tqdm only if stdout is a terminal
        for i in tqdm(range(n), disable=disable_tqdm):
            params, weights, NLL_tot = self.throw_one_from_covariance(printout)
            params_list.append(params)
            weights_list.append(weights)
            NLL_tot_list.append(NLL_tot)
            if disable_tqdm:
                if(i + 1) % n/1000 == 0:  # Print 1000 samples at regular intervals
                    print(f"Sample {i+1}/{n}: NLL = {NLL_tot:.2f}, Params = {big_vector_summary(params)}, Weights = {big_vector_summary(weights)}")
        return params_list, weights_list, NLL_tot_list

    def generate_dataset_dictionary(self, params_list, weights_list, NLL_tot_list):
        """
        Generate a dataset dictionary from the lists of parameters, weights, and NLL values, plus the covariance matrix and the best fit parameter values.
        Dictionary structure:
        {
            "data": parameter values in the real space (params_list) [N,711]
            "log_p": negative-log likelihood (NLL_tot_list) [N,1]
            "log_q": negative-log sampling probability (sum_weights_list) [N,1]
            "cov": post-fit covariance matrix (self.postfit_covariance_matrix) [711,711]
            "mean": parameter values at best-fit (self.postfit_parameter_values)  [1,711]
            "par_names": names of the parameters (self.get_parameter_names) [1,711]
            "bestfit_nll": negative-log likelihood at best fit (self.likelihood_at_bestfit) [1,1]
        }
        """
        if self.postfit_covariance_matrix is None:
            raise RuntimeError("Postfit covariance matrix is not set. Please load it first.")
        if self.postfit_parameter_values is None:
            raise RuntimeError("Postfit parameter values are not set. Please load them first.")
        if len(params_list) != len(weights_list) or len(params_list) != len(NLL_tot_list):
            raise ValueError("The lengths of params_list, weights_list, and NLL_tot_list must be the same.")

        # printout shape of all lists
        print(f"data shape: {len(params_list)} x {len(params_list[0])} ")
        log_q = [sum(weights) for weights in weights_list]
        print(f"log_q shape: {len(log_q)} ")
        print(f"log_p shape: {len(NLL_tot_list)} ")
        print(f"cov shape: {len(self.postfit_covariance_matrix)} x {len(self.postfit_covariance_matrix[0])} ")
        print(f"mean shape: {len(self.postfit_parameter_values)} ")
        print(f"par_names shape: {len(self.get_parameter_names())} ")

        dataset_dict = {
            "data": params_list,
            "log_p": NLL_tot_list,
            "log_q": log_q,
            "cov": self.postfit_covariance_matrix,
            "mean": self.postfit_parameter_values,
            "par_names": self.get_parameter_names(),
            "bestfit_nll": self.likelihood_at_bestfit
        }
        return dataset_dict

