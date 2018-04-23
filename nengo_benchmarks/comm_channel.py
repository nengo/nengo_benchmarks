import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("comm_channel")
class CommunicationChannel(object):
    """
    Nengo Benchmark Model: Communication Channel

    Input: Randomly chosen D-dimensional value
    Ouput: the same value as the input

    Parameters
    ----------
    n_neurons : int
        Number of neurons per layer
    dimensions : int
        Number of dimensions
    layers : int
        Number of layers
    pstc : float
        Synaptic time constant
    sim_time : float
        Simulation time
    """

    def __init__(self, n_neurons=100, dimensions=2, layers=2, pstc=0.01,
                 sim_time=1.0):
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.layers = layers
        self.pstc = pstc
        self.sim_time = sim_time

    def model(self):
        model = nengo.Network()
        with model:
            value = np.random.randn(self.dimensions)
            value /= np.linalg.norm(value)

            stim = nengo.Node(value)

            layers = [nengo.Ensemble(self.n_neurons, self.dimensions)
                      for _ in range(self.layers)]

            nengo.Connection(stim, layers[0], synapse=None)
            for i in range(self.layers - 1):
                nengo.Connection(layers[i], layers[i + 1], synapse=self.pstc)

            self.p_input = nengo.Probe(stim)
            self.p_output = nengo.Probe(layers[-1], synapse=self.pstc)
        return model

    def evaluate(self, sim, plt=None):
        start = timeit.default_timer()
        sim.run(self.sim_time)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        ideal = sim.data[self.p_input]
        for i in range(self.layers):
            ideal = nengo.Lowpass(self.pstc).filt(ideal, dt=sim.dt, y0=0)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.p_output])
            plt.gca().set_color_cycle(None)
            plt.plot(sim.trange(), ideal, ls='--')
            plt.ylim(-1, 1)

        rmse = np.sqrt(np.mean((sim.data[self.p_output] - ideal) ** 2))
        return dict(rmse=rmse, speed=speed)
