"""
Nengo Benchmark Model: Lorenz Attractor

Input: none
Ouput: the 3 state variables for the classic Lorenz attractor
"""

import pytry
import nengo
import numpy as np

class Lorenz(pytry.NengoTrial):
    def params(self):
        self.param('number of neurons', N=2000)
        self.param('post-synaptic time constant', tau=0.1)
        self.param('Lorenz variable', sigma=10.0)
        self.param('Lorenz variable', beta=8.0/3)
        self.param('Lorenz variable', rho=28.0)
        self.param('time to run simulation', T=10.0)

    def model(self, p):
        model = nengo.Network()
        with model:
            state = nengo.Ensemble(p.N, 3, radius=30)

            def feedback(x):
                dx0 = -p.sigma * x[0] + p.sigma * x[1]
                dx1 = -x[0] * x[2] - x[1]
                dx2 = x[0] * x[1] - p.beta * (x[2] + p.rho) - p.rho
                return [dx0 * p.tau + x[0],
                        dx1 * p.tau + x[1],
                        dx2 * p.tau + x[2]]
            nengo.Connection(state, state, function=feedback, synapse=p.tau)

            self.pState = nengo.Probe(state, synapse=p.tau)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.pState])

        return dict(
            mean=np.mean(sim.data[self.pState], axis=0).mean(),
            std=np.std(sim.data[self.pState], axis=0).mean(),
        )
