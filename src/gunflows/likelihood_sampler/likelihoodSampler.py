

import GUNDAM
import ROOT
import argparse
from pygundam_utils import *


class LikelihoodSampler:
    def __init__(self, config_file, override_files=None, threads=1, data_is_asimov=False):
        self.likelihood_interface = None
        self.cb = None
        self.cr = None
        self.fitter = None
        self.propagator = None
        self.fitter_root_file = None
        self.data_is_asimov = False  # Set to True if using asimov
        self.config_file = config_file
        self.override_files = override_files if override_files else []

        # GUNDAM.setNumberOfThreads(threads)
        # GUNDAM.setLightOutputMode(True)

        self.app = GUNDAM.GundamApp("GUNDAM: likelihood sampler")

        # read config from config file (.yaml) or Fitter output file (.root)
        if config_file.endswith('.yaml'):
            self.configure_using_yaml()
        elif config_file.endswith('.root'):
            self.configure_using_root()
        else:
            raise ValueError("Unsupported config file format. Use .yaml or .root")

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

        # Load the postfit covaraince matrix into the propagator
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
        print(f"Parameters at prior:{big_vector_summary(self.get_current_parameter_values())}")
        # Print ut parameters at best fit point


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

    def load_data_histograms(self, data_is_asimov=False):
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
            for i in range(n_bins_data):
                bin_content = data_histogram.GetBinContent(i+1)
                bin_error = data_histogram.GetBinError(i+1)
                current_bin_content = bin_content_list[i].sumWeights
                current_bin_error = bin_content_list[i].sqrtSumSqWeights
                print(f"DEBUG:bin {i}: {current_bin_content:.2f} -> {bin_content:.2f} | {current_bin_error:.2f} -> {bin_error:.2f}")
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

    def throw_n_from_covariance(self, n, printout=False):
        if self.propagator is None:
            raise RuntimeError("The propagator object is not initialized.")
        if self.likelihood_interface is None:
            raise RuntimeError("The likelihood interface is not initialized.")
        # Set custom thrower!
        self.propagator.getParametersManager().setThrowerAsCustom()

        for i in range(n):
            params, weights, NLL_tot = self.throw_one_from_covariance(printout)
