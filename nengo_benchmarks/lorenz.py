import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("lorenz")
class Lorenz(object):
    """
    Nengo Benchmark Model: Lorenz Attractor

    Input: none
    Ouput: the 3 state variables for the classic Lorenz attractor
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons
    tau : float
        Post-synaptic time constant
    sigma : float
        Lorenz variable
    beta : float
        Lorenz variable
    rho : float
        Lorenz variable
    sim_time : float
        Time to run simulation
    """

    def __init__(self, n_neurons=2000, tau=0.1, sigma=10.0, beta=8.0 / 3,
                 rho=28.0, sim_time=10.0):
        self.n_neurons = n_neurons
        self.tau = tau
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.sim_time = sim_time

    def model(self):
        model = nengo.Network()
        with model:
            state = nengo.Ensemble(self.n_neurons, 3, radius=30)

            def feedback(x):
                dx0 = -self.sigma * x[0] + self.sigma * x[1]
                dx1 = -x[0] * x[2] - x[1]
                dx2 = x[0] * x[1] - self.beta * (x[2] + self.rho) - self.rho
                return [dx0 * self.tau + x[0],
                        dx1 * self.tau + x[1],
                        dx2 * self.tau + x[2]]

            nengo.Connection(state, state, function=feedback, synapse=self.tau)

            self.p_state = nengo.Probe(state, synapse=self.tau)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        start = timeit.default_timer()
        sim.run(self.sim_time, **kwargs)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.p_state])

        return dict(
            mean=np.mean(sim.data[self.p_state], axis=0).mean(),
            std=np.std(sim.data[self.p_state], axis=0).mean(),
            speed=speed,
        )
