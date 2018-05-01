import numpy as np
import nengo
import nengo.spa as spa
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("parse")
class Parsing(object):
    """
    Nengo Benchmark Model #6: Simple Parsing

    The model parses and executes simple commands sequentially presented to it

    Parameters
    ----------
    dimensions : int
        Dimensionality of vectors
    time_per_word : float
        Time per word
    """

    def __init__(self, dimensions=32, time_per_word=0.5):
        self.dimensions = dimensions
        self.time_per_word = time_per_word

    def model(self):
        model = spa.SPA()
        with model:
            model.vision = spa.Buffer(dimensions=self.dimensions)
            model.phrase = spa.Buffer(dimensions=self.dimensions)
            model.motor = spa.Buffer(dimensions=self.dimensions)
            model.noun = spa.Memory(dimensions=self.dimensions, synapse=0.1)
            model.verb = spa.Memory(dimensions=self.dimensions, synapse=0.1)

            model.bg = spa.BasalGanglia(spa.Actions(
                'dot(vision, WRITE) --> verb=vision',
                'dot(vision, ONE+TWO+THREE) --> noun=vision',
                '0.5*(dot(vision, NONE-WRITE-ONE-TWO-THREE) + '
                'dot(phrase, WRITE*VERB))'
                '--> motor=phrase*~NOUN',
            ))
            model.thal = spa.Thalamus(model.bg)

            model.cortical = spa.Cortical(spa.Actions(
                'phrase=noun*NOUN',
                'phrase=verb*VERB',
            ))

            def vision_input(t):
                index = int(t / self.time_per_word) % 3
                return ['WRITE', 'ONE', 'NONE'][index]

            model.input = spa.Input(vision=vision_input)

            self.motor_vocab = model.get_output_vocab('motor')
            self.p_thal = nengo.Probe(model.thal.actions.output, synapse=0.03)
            self.p_motor = nengo.Probe(model.motor.state.output, synapse=0.03)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        T = self.time_per_word * 3
        start = timeit.default_timer()
        sim.run(T, **kwargs)
        end = timeit.default_timer()
        speed = T / (end - start)

        data = self.motor_vocab.dot(sim.data[self.p_motor].T).T
        mean = np.mean(data[int(self.time_per_word * 2.5 / sim.dt):], axis=0)
        correct_index = self.motor_vocab.keys.index('ONE')
        mag_correct = mean[correct_index]
        mag_others = np.mean(np.delete(mean, [correct_index]))
        mag_second = np.max(np.delete(mean, [correct_index]))

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.p_thal])
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), data)
            plt.plot(sim.trange(), data[:, correct_index], lw=2)

        return dict(mag_correct=mag_correct, mag_others=mag_others,
                    mag_second=mag_second,
                    speed=speed)
