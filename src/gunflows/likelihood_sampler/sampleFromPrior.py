import GUNDAM
import ROOT
import argparse
def get_parameter_values(lh_interface):
    """
    Helper function to extract parameter values from the likelihood interface.
    """
    propagator = lh_interface.getModelPropagator()
    parameter_sets_list = propagator.getParametersManager().getParameterSetsList()
    parameter_values = []
    for parameterSet in parameter_sets_list:
        if not parameterSet.isEnabled():
            continue
        for parameter in parameterSet.getParameterList():
            if not parameter.isEnabled():
                continue
            parameter_values.append(float(parameter.getParameterValue()))
    return parameter_values



GUNDAM.setNumberOfThreads(2)
GUNDAM.setLightOutputMode(True)

app = GUNDAM.GundamApp("GUNDAM: sample LH from prior covariance matrix")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', required=True, help='Config file path')
    parser.add_argument('-n', default=1, type=int, help='Number of toys')
    parser.add_argument('-of', nargs='+', help='Override config file paths')
    parser.add_argument('-o', required=True, help='Output file path')
    args = parser.parse_args()

    print("Running nToys=", args.n)
    nToys = args.n

    print("Using base config file:", args.c)
    cb = GUNDAM.ConfigUtils.ConfigBuilder(args.c)
    if (args.of is not None):
        for override_file in args.of:
            print("Using override:", override_file)
            cb.override(override_file)

    # Config reader setup:
    cr = GUNDAM.ConfigUtils.ConfigReader(cb.getConfig())
    cr.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    fitterEngineConfig = cr.fetchValueConfigReader("fitterEngineConfig")

    app.openOutputFile(args.o)
    app.writeAppInfo()

    # Fitter setup:
    fitter = GUNDAM.FitterEngine()
    fitter.setConfig(fitterEngineConfig)
    fitter.configure()

    # Set the data as asimov (prior)
    fitter.getLikelihoodInterface().setForceAsimovData(True)

    # Do I need this?
    # fitter.getLikelihoodInterface().getModelPropagator().setEnableEigenToOrigInPropagate( false );

    # Initialize the fitter engine
    fitter.initialize()

    ######################################
    #        INITIALIZATION DONE         #
    ######################################

    # Print out NLL at prior (should be 0 ...)
    # Get the likelihood interface
    lh_interface = fitter.getLikelihoodInterface()

    # propagate parameters to compute likelihood
    lh_interface.propagateAndEvalLikelihood()

    prior_stat_nll = lh_interface.getBuffer().statLikelihood
    prior_syst_nll = lh_interface.getBuffer().penaltyLikelihood

    print("Prior stat NLL:", prior_stat_nll)
    print("Prior syst NLL:", prior_syst_nll)

    # Print out the parameters configuration
    print("Prior parameters configuration:")
    propagator = lh_interface.getModelPropagator()
    parameterSetsList = propagator.getParametersManager().getParameterSetsList()
    n_enabled_parameters = 0
    parameter_values = []
    for parameterSet in parameterSetsList:
        print("Parameter set:", parameterSet.getName())
        if not parameterSet.isEnabled():
            print("  disabled")
            continue
        n_enabled_parameters_this_set = 0
        for parameter in parameterSet.getParameterList():
            if not parameter.isEnabled() :
                print(f"  {parameter.getFullTitle()} disabled")
                continue
            if parameter.getPhysicalLimits().hasBound:
                minBound = parameter.getPhysicalLimits().min
                maxBound = parameter.getPhysicalLimits().max
                limits = f"limits: [{minBound:.3f}, {maxBound:.3f}]"
            else:
                limits = "no limits"
            print(f"  {parameter.getFullTitle()}: {parameter.getParameterValue():.3f} - prior: {parameter.getPriorValue():.3f}, sigma: {parameter.getStdDevValue():.3f}, {limits}")
            parameter_values.append(parameter.getParameterValue())
            n_enabled_parameters += 1
            n_enabled_parameters_this_set += 1
        print(f"  Number of enabled parameters in set {parameterSet.getName()}: {n_enabled_parameters_this_set}")
    print("Total number of enabled parameters:", n_enabled_parameters)

    # Throw parameters
    # print parameters at prior
    display_values = [f"{v:.3f}" for v in parameter_values[:5]] + ['..'] + [f"{v:.3f}" for v in parameter_values[-5:]]
    print("prior parameters")
    print(display_values)

    # Necessary to use the thrower that saves the weights
    propagator.getParametersManager().setThrowerAsCustom()

    for iToy in range(nToys):
        weights = []
        propagator.getParametersManager().throwParametersFromGlobalCovariance(weights)
        parameter_values = get_parameter_values(lh_interface)
        print(f"Toy {iToy} parameters after throw:")
        display_values = [f"{v:.3f}" for v in parameter_values[:5]] + ['..'] + [f"{v:.3f}" for v in parameter_values[-5:]]
        print(display_values)



































if __name__ == "__main__":
    main()
    exit(0)