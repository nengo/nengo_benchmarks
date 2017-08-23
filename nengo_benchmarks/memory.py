"""
Nengo SPA Benchmark Model: Semantic Memory

The model remembers and attempts to recall a sequence of bound symbols

"""

import numpy as np
import nengo
import nengo.spa as spa

import pytry
import timeit

class SemanticMemory(pytry.NengoTrial):
    def params(self):
        self.param('time per symbol', time_per_symbol=0.2)
        self.param('number of symbols', n_symbols=4)
        self.param('time to recall', T=1.0)
        self.param('dimensions', D=16)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.word = spa.State(dimensions=p.D)
            model.marker = spa.State(dimensions=p.D)
            model.memory = spa.State(dimensions=p.D, feedback=1)

            model.cortical = spa.Cortical(spa.Actions(
                'memory = word * marker',
                ))

            def word(t):
                index = t / p.time_per_symbol
                if index < p.n_symbols:
                    return 'S%d' % index
                return '0'

            def marker(t):
                index = t / p.time_per_symbol
                if index < p.n_symbols:
                    return 'M%d' % index
                return '0'

            model.input = spa.Input(word=word, marker=marker)

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
            self.vocab = model.get_output_vocab('memory')

        return model

    def evaluate(self, p, sim, plt):
        T = p.T + p.time_per_symbol * p.n_symbols
        start = timeit.default_timer()
        sim.run(T)
        end = timeit.default_timer()
        speed = T / (end - start)

        pairs = np.zeros((p.n_symbols, p.D), dtype=float)
        for i in range(p.n_symbols):
            pairs[i] = self.vocab.parse('S%d*M%d' % (i, i)).v

        data_memory = sim.data[self.p_memory]
        memory = np.dot(pairs, data_memory.T).T

        times = sim.trange()

        mean_memory = np.mean(memory[-1])

        if plt is not None:
            plt.plot(sim.trange(), memory)


        return dict(memory = mean_memory,
                    speed=speed)
