"""
Nengo Benchmark Model: Circular Convolution

Input: two random D-dimensional vectors
Output: the circular convolution of the inputs

"""


import benchmark
import nengo
import nengo.spa as spa
import numpy as np

class CircularConvolution(benchmark.Benchmark):
    def params(self):
        return dict(
            D=8,      # dimensionality
            T=0.5,    # time to run the simulation for
            SD=8,     # subdimensions
            pstc=0.01 # synapse filter
            )
    def benchmark(self, p, Simulator, rng, plt):
        model = spa.SPA(seed=p.seed)
        with model:
            model.inA = spa.Buffer(p.D, subdimensions=p.SD)
            model.inB = spa.Buffer(p.D, subdimensions=p.SD)

            model.result = spa.Buffer(p.D, subdimensions=p.SD)

            model.cortical = spa.Cortical(spa.Actions('result = inA * inB'),
                                          synapse=p.pstc)

            model.input = spa.Input(inA='A', inB='B')

            probe = nengo.Probe(model.result.state.output, synapse=p.pstc)

            ideal = nengo.Node(model.get_output_vocab('inA').parse('A*B').v)
            probe_ideal = nengo.Probe(ideal, synapse=None)

        sim = Simulator(model)
        sim.run(p.T)

        ideal = sim.data[probe_ideal]
        for i in range(3):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(0.05), p.dt)


        if plt is not None:
            plt.plot(sim.trange(), sim.data[probe])
            plt.plot(sim.trange(), ideal)


        rmse = np.sqrt(np.mean(sim.data[probe] - ideal)**2)
        return dict(rmse=rmse)

if __name__ == '__main__':
    b = CircularConvolution().run()
