"""
Nengo Benchmark Model: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

import benchmark
import nengo
import numpy as np

class CommunicationChannel(benchmark.Benchmark):
    def params(self):
        return dict(
            D=2,       # number of dimensions
            L=2,       # number of layers
            N=100,     # number of neurons per layer
            pstc=0.01, # synaptic time constant
            T=1.0,     # amount of time to run for
        )

    def benchmark(self, p, Simulator, rng, plt):
        model = nengo.Network(seed=p.seed)
        with model:
            value = rng.randn(p.D)
            value /= np.linalg.norm(value)

            input = nengo.Node(value)

            layers = [nengo.Ensemble(p.N, p.D) for i in range(p.L)]

            nengo.Connection(input, layers[0])
            for i in range(p.L-1):
                nengo.Connection(layers[i], layers[i+1], synapse=p.pstc)

            pInput = nengo.Probe(input)
            pOutput = nengo.Probe(layers[-1], synapse=p.pstc)

        import nengo_info
        info = nengo_info.NengoInfo(model)
        info.print_info()

        sim = Simulator(model, dt=p.dt)
        sim.run(p.T)

        ideal = sim.data[pInput]
        for i in range(p.L):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.pstc), p.dt)


        if plt is not None:
            plt.plot(sim.trange(), sim.data[pOutput])
            plt.plot(sim.trange(), ideal)
            plt.ylim(-1,1)

        rmse = np.sqrt(np.mean(sim.data[pOutput] - ideal)**2)
        return dict(rmse=rmse)

if __name__ == '__main__':
    b = CommunicationChannel().run()
