import inspect

import nengo
import pytest
import matplotlib.pyplot as plt

from nengo_benchmarks import all_benchmarks, run_pytry


@pytest.mark.parametrize("Benchmark", all_benchmarks.values())
@pytest.mark.parametrize("use_pytry", (True, False))
def test_all(Benchmark, use_pytry):
    # use dimensions or n_neurons as the parameter (to make sure that we can
    # pass parameters)
    params = inspect.signature(Benchmark.__init__).parameters
    kwargs = {}
    if "dimensions" in params:
        kwargs["dimensions"] = 1
    if "n_neurons" in params:
        kwargs["n_neurons"] = 8

    if len(kwargs) == 0:
        raise KeyError("Benchmark does not contain a dimensions/n_neurons "
                       "parameter")

    if use_pytry:
        run_pytry.wrap(Benchmark()).run(save_figs=True, **kwargs)
    else:
        bench = Benchmark(**kwargs)
        with nengo.Simulator(bench.model()) as sim:
            bench.evaluate(sim, plt=plt)
