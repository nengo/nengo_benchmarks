import numpy as np
import nengo
import nengo.spa as spa
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("memory")
class SemanticMemory(object):
    """
    Nengo SPA Benchmark Model: Semantic Memory

    The model remembers and attempts to recall a sequence of bound symbols

    Parameters
    ----------
    dimensions : int
        Dimensionality of vectors
    time_per_symbol : float
        Time per symbol
    n_symbols : int
        Number of symbols
    recall_time : float
        Time to recall
    """

    def __init__(self, dimensions=16, time_per_symbol=0.2, n_symbols=4,
                 recall_time=1.0):
        self.dimensions = dimensions
        self.time_per_symbol = time_per_symbol
        self.n_symbols = n_symbols
        self.recall_time = recall_time

    def model(self):
        model = spa.SPA()
        with model:
            model.word = spa.State(dimensions=self.dimensions)
            model.marker = spa.State(dimensions=self.dimensions)
            model.memory = spa.State(dimensions=self.dimensions, feedback=1)

            model.cortical = spa.Cortical(spa.Actions(
                'memory = word * marker',
            ))

            def word(t):
                index = t / self.time_per_symbol
                if index < self.n_symbols:
                    return 'S%d' % index
                return '0'

            def marker(t):
                index = t / self.time_per_symbol
                if index < self.n_symbols:
                    return 'M%d' % index
                return '0'

            model.input = spa.Input(word=word, marker=marker)

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
            self.vocab = model.get_output_vocab('memory')

        return model

    def evaluate(self, sim, plt=None, **kwargs):
        T = self.recall_time + self.time_per_symbol * self.n_symbols
        start = timeit.default_timer()
        sim.run(T, **kwargs)
        end = timeit.default_timer()
        speed = T / (end - start)

        pairs = np.zeros((self.n_symbols, self.dimensions), dtype=float)
        for i in range(self.n_symbols):
            pairs[i] = self.vocab.parse('S%d*M%d' % (i, i)).v

        data_memory = sim.data[self.p_memory]
        memory = np.dot(pairs, data_memory.T).T

        mean_memory = np.mean(memory[-1])

        if plt is not None:
            plt.plot(sim.trange(), memory)

        return dict(memory=mean_memory,
                    speed=speed)
