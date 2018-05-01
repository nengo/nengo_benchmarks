import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("learning")
class LearningSpeedup(object):
    """
    Nengo Benchmark Model: Circular Convolution
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons per ensemble
    dimensions : int
        Dimensionality of ensembles
    tau_slow : float
        Slow path time constant
    tau_fast : float
        Fast path time constant
    sim_time : float
        Time to simulate for
    n_switches : int
        Number of times to change function
    learn_rate : float
        Learning rate
    """

    def __init__(self, n_neurons=100, dimensions=1, tau_slow=0.2,
                 tau_fast=0.01, sim_time=40.0, n_switches=2, learn_rate=1e-4):
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.tau_slow = tau_slow
        self.tau_fast = tau_fast
        self.sim_time = sim_time
        self.n_switches = n_switches
        self.learn_rate = learn_rate

    def model(self):
        model = nengo.Network()

        with model:
            def stim(t):
                return [np.sin(t + i * np.pi * 2 / self.dimensions)
                        for i in range(self.dimensions)]

            pre_value = nengo.Node(stim)

            pre = nengo.Ensemble(self.n_neurons, self.dimensions)
            post = nengo.Ensemble(self.n_neurons, self.dimensions)
            target = nengo.Ensemble(self.n_neurons, self.dimensions)
            nengo.Connection(pre_value, pre, synapse=None)

            conn = nengo.Connection(
                pre, post,
                function=lambda x: np.random.random(size=self.dimensions),
                learning_rule_type=nengo.PES(learning_rate=self.learn_rate))

            slow = nengo.networks.Product(
                self.n_neurons * 2 // self.dimensions, self.dimensions)
            T_context = self.sim_time / self.n_switches
            context = nengo.Node(lambda t: 1 if int(t / T_context) % 2 else -1)

            nengo.Connection(
                context, slow.A, transform=np.ones((self.dimensions, 1)))

            nengo.Connection(pre, slow.B, synapse=self.tau_slow)

            nengo.Connection(slow.output, target, synapse=self.tau_slow)

            error = nengo.Ensemble(n_neurons=self.n_neurons,
                                   dimensions=self.dimensions)

            nengo.Connection(post, error,
                             synapse=self.tau_slow * 2 + self.tau_fast)
            nengo.Connection(target, error, transform=-1,
                             synapse=self.tau_fast)

            nengo.Connection(error, conn.learning_rule)

            self.probe_target = nengo.Probe(target, synapse=self.tau_fast)
            self.probe_post = nengo.Probe(post, synapse=self.tau_fast)
            self.probe_pre = nengo.Probe(pre_value, synapse=None)
            self.probe_context = nengo.Probe(context, synapse=None)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        start = timeit.default_timer()
        sim.run(self.sim_time, **kwargs)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        ideal = sim.data[self.probe_pre] * sim.data[self.probe_context]
        for i in range(2):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(self.tau_fast),
                                        self.dimensions)

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.probe_target])
            plt.plot(sim.trange(), ideal)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), sim.data[self.probe_post])
            plt.plot(sim.trange(), ideal)

        rmse = np.sqrt(np.mean((sim.data[self.probe_post] - ideal) ** 2))
        return dict(rmse=rmse, speed=speed)
