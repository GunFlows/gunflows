import PyGundam

import argparse

import ROOT 


PyGundam.setNumberOfThreads(2)
PyGundam.setLightOutputMode(True)

app = PyGundam.GundamApp("PyGundam pull toy script")


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
    cb = PyGundam.ConfigUtils.ConfigBuilder(args.c)
    for override_file in args.of:
        print("Using override:", override_file)
        cb.override(override_file)

    cr = PyGundam.ConfigUtils.ConfigReader(cb.getConfig())
    cr.defineField(PyGundam.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    fitterEngineConfig = cr.fetchValueConfigReader("fitterEngineConfig")

    app.openOutputFile(args.o)
    app.writeAppInfo()

    e = PyGundam.FitterEngine()
    e.setConfig(fitterEngineConfig)
    e.configure()

    e.getLikelihoodInterface().setForceAsimovData(True)

    e.initialize()

    l = e.getLikelihoodInterface()

    priorParameters = None

    for iToy in range(nToys):
        e.setSaveDir(app, f"toy_{iToy}")

        if priorParameters is None:
            priorParameters = l.getModelPropagator().getParametersManager().exportParameterInjectorConfig()
        else:
            l.getModelPropagator().getParametersManager().injectParameterValues(priorParameters)
            # restoring the original prior
            l.setCurrentParameterValuesAsPrior()
            # restoring state of the model before syst throws
            l.propagateAndEvalLikelihood()
            # restoring the state of the data before statistical throws
            l.getDataPropagator().copyHistBinContentFrom(l.getModelPropagator())

        l.throwStatErrors(l.getDataPropagator())
        l.throwToyParameters(l.getModelPropagator())
        l.setCurrentParameterValuesAsPrior()

        l.propagateAndEvalLikelihood()
        print("Initial likelihood summary:", l.getSummary())

        print("Starting the fit")
        e.fit()


if __name__ == "__main__":
    main()
    exit(0)
