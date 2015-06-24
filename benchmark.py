import argparse
import importlib
import os

import matplotlib.pyplot
import numpy as np


class Benchmark(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Nengo benchmark')
        parser.add_argument('--no_figs', action='store_true')
        parser.add_argument('--show_figs', action='store_true')
        parser.add_argument('--subdir', type=str, default='data')

        params = self.params()

        if 'backend' not in params:
            params['backend'] = 'nengo'
        if 'seed' not in params:
            params['seed'] = 1
        if 'dt' not in params:
            params['dt'] = 0.001

        for k, v in params.items():
            parser.add_argument('--%s' % k, type=type(v), default=v)

        args = parser.parse_args()

        name = self.__class__.__name__
        text = []
        for k in sorted(params.keys()):
            text.append('%s=%s' % (k, getattr(args, k)))

        self.filename = name + '#' + ','.join(text)

        self.param_settings = args

    def run(self):
        print('running %s' % self.filename)
        p = self.param_settings
        rng = np.random.RandomState(seed=p.seed)
        module = importlib.import_module(p.backend)
        Simulator = module.Simulator
        if not p.no_figs or p.show_figs:
            plt = matplotlib.pyplot
        else:
            plt = None
        result = self.benchmark(p, Simulator, rng, plt)

        text = []
        for k, v in sorted(result.items()):
            text.append('%s = %s' % (k, repr(v)))
        text = '\n'.join(text)


        if plt is not None:
            plt.title(self.filename.replace('#', '\n') +'\n' + text,
                      fontsize=10)

        fn = self.filename
        if not os.path.exists(p.subdir):
            os.mkdir(p.subdir)
        fn = os.path.join(p.subdir, fn)
        if not p.no_figs:
            plt.savefig(fn + '.png')
        if p.show_figs:
            plt.show()

        with open(fn + '.txt', 'w') as f:
            f.write(text)
        print(text)


