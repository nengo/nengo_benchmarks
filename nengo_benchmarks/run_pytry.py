import inspect
import sys

from numpydoc.docscrape import NumpyDocString
import pytry

import nengo_benchmarks


def wrap(obj):
    class PytryWrapper(pytry.NengoTrial):
        def params(self):
            doc = {k: v[0] for k, _, v in
                   NumpyDocString(obj.__doc__)["Parameters"]}

            for p in inspect.signature(obj.__init__).parameters:
                self.param(doc[p], **{p: getattr(obj, p)})

        def model(self, p):
            for param in self.param_defaults:
                setattr(obj, param, getattr(p, param))
            return obj.model()

        def evaluate(self, p, sim, plt):
            for param in self.param_defaults:
                setattr(obj, param, getattr(p, param))

            return obj.evaluate(sim, plt)

    wrapped = PytryWrapper()

    return wrapped


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Must specify name of trial to be run")
    benchmark = sys.argv[1]

    if benchmark not in nengo_benchmarks.all_benchmarks:
        raise KeyError("%s is not a known benchmark" % benchmark)
    trial = wrap(nengo_benchmarks.all_benchmarks[benchmark]())

    trial.run(**pytry.parser.parse_args(trial, args=None, allow_filename=True))
