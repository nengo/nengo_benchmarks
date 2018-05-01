import nengo
import nengo.spa as spa
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("convolution")
class CircularConvolution(object):
    """
    Nengo Benchmark Model: Circular Convolution

    Input: two random D-dimensional vectors
    Output: the circular convolution of the inputs

    Parameters
    ----------
    n_neurons : int
        Neurons per circular convolution
    dimensions : int
        Dimensionality of input/output vectors
    sim_time : float
        Time to run simulation
    subdimensions : int
        Dimensionality of sub-populations
    pstc : float
        Post-synaptic time constant
    n_neurons_io : int
        Neurons per input/output buffer
    """

    def __init__(self, n_neurons=1600, dimensions=8, sim_time=0.5,
                 subdimensions=8, pstc=0.01, n_neurons_io=400):
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.sim_time = sim_time
        self.subdimensions = subdimensions
        self.pstc = pstc
        self.n_neurons_io = n_neurons_io

    def model(self):
        model = spa.SPA()
        with model:
            model.inA = spa.Buffer(
                self.dimensions, subdimensions=self.subdimensions,
                neurons_per_dimension=self.n_neurons_io // self.dimensions)
            model.inB = spa.Buffer(
                self.dimensions, subdimensions=self.subdimensions,
                neurons_per_dimension=self.n_neurons_io // self.dimensions)

            model.result = spa.Buffer(
                self.dimensions, subdimensions=self.subdimensions,
                neurons_per_dimension=self.n_neurons_io // self.dimensions)

            model.cortical = spa.Cortical(
                spa.Actions('result = inA * inB'), synapse=self.pstc,
                neurons_cconv=self.n_neurons // self.dimensions)

            model.input = spa.Input(inA='A', inB='B')

            self.probe = nengo.Probe(model.result.state.output,
                                     synapse=self.pstc)

            ideal = nengo.Node(model.get_output_vocab('inA').parse('A*B').v)
            self.probe_ideal = nengo.Probe(ideal, synapse=None)

        return model

    def evaluate(self, sim, plt=None, **kwargs):
        start = timeit.default_timer()
        sim.run(self.sim_time, **kwargs)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        ideal = sim.data[self.probe_ideal]
        for i in range(3):
            ideal = nengo.Lowpass(self.pstc).filt(ideal, dt=sim.dt, y0=0)

        # compute where to check results from
        index = int(self.pstc * 3 * 4 / sim.dt)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.gca().set_color_cycle(None)
            plt.plot(sim.trange(), ideal, ls='--')
            plt.axvline(index * sim.dt, c='#aaaaaa')

        rmse = np.sqrt(
            np.mean((sim.data[self.probe][index:] - ideal[index:]) ** 2))
        return dict(rmse=rmse, speed=speed)
