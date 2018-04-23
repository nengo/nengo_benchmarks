import numpy as np
import nengo
import nengo.spa as spa
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("sequence")
class SPASequence(object):
    """
    Nengo Benchmark Model: SPA Sequence

    Given no input, the model will cycle between cortical states using a
    basal ganglia and thalamus.
    
    Parameters
    ----------
    dimensions : int
        Dimensionality of vectors
    n_actions : int
        Number of actions
    sim_time : float
        Time to simulate
    """

    def __init__(self, dimensions=32, n_actions=5, sim_time=1.0):
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.sim_time = sim_time

    def model(self):
        model = spa.SPA()
        with model:
            model.state = spa.Memory(dimensions=self.dimensions)
            actions = [
                'dot(state, S%d) --> state=S%d' % (i, (i + 1) % self.n_actions)
                for i in range(self.n_actions)]
            model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
            model.thal = spa.Thalamus(model.bg)

            def state_input(t):
                if t < 0.1:
                    return 'S0'
                else:
                    return '0'

            model.input = spa.Input(state=state_input)

            self.probe = nengo.Probe(model.thal.actions.output, synapse=0.03)
        return model

    def evaluate(self, sim, plt=None):
        start = timeit.default_timer()
        sim.run(self.sim_time)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        index = int(0.05 / sim.dt)  # ignore the first 50ms
        best = np.argmax(sim.data[self.probe][index:], axis=1)
        change = np.diff(best)
        change_points = np.where(change != 0)[0]
        intervals = np.diff(change_points * sim.dt)

        data = sim.data[self.probe][index:]
        peaks = [np.max(data[change_points[i]:change_points[i + 1]])
                 for i in range(len(change_points) - 1)]

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.plot(sim.trange()[index + 1:], np.where(change != 0, 1, 0))
            for i, peak in enumerate(peaks):
                plt.plot([sim.dt * (change_points[i] + index),
                          sim.dt * (change_points[i + 1] + index)],
                         [peak, peak], color='b')

        return dict(period=np.mean(intervals), period_sd=np.std(intervals),
                    peak=np.mean(peaks), peak_sd=np.std(peaks),
                    speed=speed)
