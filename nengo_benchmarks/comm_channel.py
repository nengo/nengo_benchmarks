"""
Nengo Benchmark Model: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

import nengo
import numpy as np
import pytry
import timeit

class CommunicationChannel(pytry.NengoTrial):
    def params(self):
        self.param('number of dimensions', D=2)
        self.param('number of layers', L=2)
        self.param('number of neurons per layer', N=100)
        self.param('synaptic time constant', pstc=0.01)
        self.param('simulation time', T=1.0)

    def model(self, p):
        model = nengo.Network()
        with model:
            value = np.random.randn(p.D)
            value /= np.linalg.norm(value)

            stim = nengo.Node(value)

            layers = [nengo.Ensemble(p.N, p.D) for i in range(p.L)]

            nengo.Connection(stim, layers[0], synapse=None)
            for i in range(p.L-1):
                nengo.Connection(layers[i], layers[i+1], synapse=p.pstc)

            self.pInput = nengo.Probe(stim)
            self.pOutput = nengo.Probe(layers[-1], synapse=p.pstc)
        return model


    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        sim.run(p.T)
        end = timeit.default_timer()
        speed = p.T / (end - start)

        ideal = sim.data[self.pInput]
        for i in range(p.L):
            ideal = nengo.Lowpass(p.pstc).filt(ideal, dt=p.dt, y0=0)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.pOutput])
            plt.gca().set_color_cycle(None)
            plt.plot(sim.trange(), ideal, ls='--')
            plt.ylim(-1, 1)

        rmse = np.sqrt(np.mean((sim.data[self.pOutput] - ideal)**2))
        return dict(rmse=rmse, speed=speed)
