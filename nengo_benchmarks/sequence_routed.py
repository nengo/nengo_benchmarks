import numpy as np
import nengo
import nengo.spa as spa
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("sequence_routed")
class SPASequenceRouted(object):
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
    start : int
        Starting action
    """

    def __init__(self, dimensions=32, n_actions=5, sim_time=1.0, start=0):
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.sim_time = sim_time
        self.start = start

    def model(self):
        model = spa.SPA()
        with model:
            model.vision = spa.Buffer(dimensions=self.dimensions)
            model.state = spa.Memory(dimensions=self.dimensions)
            actions = ['dot(state, S%d) --> state=S%d' % (i, (i + 1))
                       for i in range(self.n_actions - 1)]
            actions.append('dot(state, S%d) --> state=vision' %
                           (self.n_actions - 1))
            model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
            model.thal = spa.Thalamus(model.bg)

            model.input = spa.Input(vision='S%d' % self.start,
                                    state=lambda t: 'S%d' % self.start if
                                    t < 0.1 else '0')

            self.probe = nengo.Probe(model.thal.actions.output, synapse=0.03)

        return model

    def evaluate(self, sim, plt=None):
        start = timeit.default_timer()
        sim.run(self.sim_time)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        index = int(0.05 / sim.dt)  # ignore the first 50ms
        best = np.argmax(sim.data[self.probe][index:], axis=1)
        times = sim.trange()
        change = np.diff(best)
        change_points = np.where(change != 0)[0]
        intervals = np.diff(change_points * sim.dt)

        best_index = best[change_points][1:]
        route_intervals = intervals[
            np.where(best_index == self.n_actions - 1)[0]]
        seq_intervals = intervals[
            np.where(best_index != self.n_actions - 1)[0]]

        data = sim.data[self.probe][index:]
        peaks = [np.max(data[change_points[i]:change_points[i + 1]])
                 for i in range(len(change_points) - 1)]

        if plt is not None:
            plt.plot(times, sim.data[self.probe])
            plt.plot(times[index + 1:], np.where(change != 0, 1, 0))

            for i, peak in enumerate(peaks):
                plt.hlines(peak, times[change_points[i] + index],
                           times[change_points[i + 1] + index])

        return dict(period=np.mean(seq_intervals),
                    period_sd=np.std(seq_intervals),
                    route_period=np.mean(route_intervals),
                    route_period_sd=np.std(route_intervals),
                    peak=np.mean(peaks), peak_sd=np.std(peaks),
                    speed=speed)
