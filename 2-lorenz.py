"""
Nengo Benchmark Model: Lorenz Attractor

Input: none
Ouput: the 3 state variables for the classic Lorenz attractor
"""

import benchmark
import nengo
import numpy as np

class Lorenz(benchmark.Benchmark):
    def params(self):
        return dict(
            N=2000,      # number of neurons
            tau=0.1,     # post-synaptic time constant
            sigma=10,    # Lorenz variables
            beta=8.0/3,  # Lorenz variables
            rho=28,      # Lorenz variables
            T=10.0,      # time to run the simulation for
        )
    def benchmark(self, p, Simulator, rng, plt):
        model = nengo.Network(seed=p.seed)
        with model:
            state = nengo.Ensemble(p.N, 3, radius=60)

            def feedback(x):
                dx0 = -p.sigma * x[0] + p.sigma * x[1]
                dx1 = -x[0] * x[2] - x[1]
                dx2 = x[0] * x[1] - p.beta * (x[2] + p.rho) - p.rho
                return [dx0 * p.tau + x[0],
                        dx1 * p.tau + x[1],
                        dx2 * p.tau + x[2]]
            nengo.Connection(state, state, function=feedback, synapse=p.tau)

            pState = nengo.Probe(state, synapse=p.tau)

        sim = Simulator(model, dt=p.dt)
        sim.run(p.T)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[pState])

        return dict(
            mean=np.mean(sim.data[pState], axis=0).mean(),
            std=np.std(sim.data[pState], axis=0).mean(),
        )


if __name__ == '__main__':
    b = Lorenz().run()
