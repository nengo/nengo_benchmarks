import numpy as np
import nengo
import nengo.spa as spa
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("memory_recall")
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
    time_per_cue : float
        Time per cue
    n_symbols : int
        Number of symbols
    recall_time : float
        Time to recall
    """

    def __init__(self, dimensions=16, time_per_symbol=0.2, time_per_cue=0.1,
                 n_symbols=4, recall_time=1.0):
        self.dimensions = dimensions
        self.time_per_symbol = time_per_symbol
        self.time_per_cue = time_per_cue
        self.n_symbols = n_symbols
        self.recall_time = recall_time

    def model(self):
        model = spa.SPA()
        with model:
            model.word = spa.State(dimensions=self.dimensions)
            model.marker = spa.State(dimensions=self.dimensions)
            model.memory = spa.State(dimensions=self.dimensions, feedback=1)
            model.motor = spa.State(dimensions=self.dimensions)
            model.cue = spa.State(dimensions=self.dimensions)

            model.cortical = spa.Cortical(spa.Actions(
                'memory = word * marker',
                'motor = memory * ~cue',
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

            def cue(t):
                index = (t - self.time_per_symbol *
                         self.n_symbols) / self.time_per_cue
                if index > 0:
                    index = index % (2 * self.n_symbols)
                    if index < self.n_symbols:
                        return 'S%d' % index
                    else:
                        return 'M%d' % (index - self.n_symbols)
                return '0'

            model.input = spa.Input(word=word, marker=marker, cue=cue)

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
            self.p_motor = nengo.Probe(model.motor.output, synapse=0.03)
            self.vocab = model.get_output_vocab('motor')

        return model

    def evaluate(self, sim, plt=None):
        T = self.recall_time + self.time_per_symbol * self.n_symbols
        start = timeit.default_timer()
        sim.run(T)
        end = timeit.default_timer()
        speed = T / (end - start)

        pairs = np.zeros((self.n_symbols, self.dimensions), dtype=float)
        for i in range(self.n_symbols):
            pairs[i] = self.vocab.parse('S%d*M%d' % (i, i)).v

        data_memory = sim.data[self.p_memory]
        memory = np.dot(pairs, data_memory.T).T
        data_motor = sim.data[self.p_motor]
        motor = self.vocab.dot(data_motor.T).T

        times = sim.trange()

        mean_memory = np.mean(memory[-1])

        mag_correct = []
        mag_others = []
        mag_second = []
        index = 0
        for i in range(int(self.recall_time / self.time_per_cue)):
            t = self.time_per_symbol * self.n_symbols + (
                    i + 1) * self.time_per_cue
            while index < len(times) - 1 and times[index + 1] < t:
                index += 1
            correct = i % (2 * self.n_symbols)
            if correct < self.n_symbols:
                correct_index = self.vocab.keys.index('M%d' % correct)
            else:
                correct_index = self.vocab.keys.index(
                    'S%d' % (correct - self.n_symbols))
            mag_correct.append(motor[index, correct_index])
            mag_others.append(
                np.mean(np.delete(motor[index], [correct_index])))
            mag_second.append(np.max(np.delete(motor[index], [correct_index])))

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), memory)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), motor)

        return dict(mag_correct=np.mean(mag_correct),
                    mag_others=np.mean(mag_others),
                    mag_second=np.mean(mag_second),
                    memory=mean_memory,
                    speed=speed)
