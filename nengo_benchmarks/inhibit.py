import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("inhibit")
class Inhibition(object):
    """
    Nengo Benchmark Model: Circular Convolution

    Parameters
    ----------
    n_neurons : int
        Number of neurons
    dimensions : int
        Number of dimensions
    inh_strength : float
        Inhibition strength
    sim_time : float
        Time to run
    """

    def __init__(self, n_neurons=100, dimensions=1, inh_strength=2.0,
                 sim_time=3.0):
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.inh_strength = inh_strength
        self.sim_time = sim_time

    def model(self):
        model = nengo.Network()
        with model:
            stim = nengo.Node(1)
            ens = nengo.Ensemble(n_neurons=self.n_neurons,
                                 dimensions=self.dimensions)
            nengo.Connection(
                stim, ens, transform=np.ones((self.dimensions, 1)),
                synapse=None)

            stim_inh = nengo.Node(lambda t: t)
            inh = nengo.Ensemble(n_neurons=100, dimensions=1)
            nengo.Connection(stim_inh, inh, synapse=None)
            nengo.Connection(
                inh, ens.neurons, transform=np.ones(
                    (self.n_neurons, 1)) * -self.inh_strength,
                synapse=None)

            self.p_ens = nengo.Probe(ens, synapse=0.01)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        start = timeit.default_timer()
        sim.run(self.sim_time, **kwargs)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        data = sim.data[self.p_ens]

        last = []
        for row in data.T:
            nz = np.nonzero(row > 0.05)[0]
            if len(nz) == 0:
                last.append(0)
            else:
                last.append(nz[-1])
        time_to_inhibit = np.array(last) * sim.dt

        if plt:
            plt.plot(sim.trange(), sim.data[self.p_ens])
            for t in time_to_inhibit:
                plt.axvline(t)
            plt.axhline(0.05, linestyle='--', c='k')

        return dict(time_to_inhibit=np.mean(time_to_inhibit),
                    speed=speed)
