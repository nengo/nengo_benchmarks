import nengo
import nengo.spa as spa
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("conv_cleanup")
class ConvolutionCleanup(object):
    """
    Nengo Benchmark Model: Convolution Cleanup

    Parameters
    ----------
    n_neurons : int
        Number of neurons per circular convolution
    dimensions : int
        Dimensionality of input/output vectors
    mem_tau : float
        Memory time constant
    mem_input_scale : float
        Input scaling on memory
    test_time : float
        Amount of time to test memory for
    test_present_time : float
        Amount of time per test
    """

    def __init__(self, n_neurons=3200, dimensions=16, mem_tau=0.1,
                 mem_input_scale=0.5, test_time=2.0, test_present_time=0.1):
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.mem_tau = mem_tau
        self.mem_input_scale = mem_input_scale
        self.test_time = test_time
        self.test_present_time = test_present_time

    def model(self):
        model = spa.SPA()
        with model:
            model.shape = spa.Buffer(self.dimensions)
            model.color = spa.Buffer(self.dimensions)
            model.bound = spa.Buffer(self.dimensions)

            cconv = nengo.networks.CircularConvolution(
                n_neurons=self.n_neurons // self.dimensions,
                dimensions=self.dimensions)

            nengo.Connection(model.shape.state.output, cconv.A)
            nengo.Connection(model.color.state.output, cconv.B)
            nengo.Connection(
                cconv.output, model.bound.state.input,
                transform=self.mem_input_scale, synapse=self.mem_tau)

            deconv = nengo.networks.CircularConvolution(
                n_neurons=self.n_neurons, dimensions=self.dimensions,
                invert_b=True)
            deconv.label = 'deconv'

            model.query = spa.Buffer(self.dimensions)
            model.result = spa.Buffer(self.dimensions)

            nengo.Connection(model.bound.state.output, deconv.A)
            nengo.Connection(model.query.state.output, deconv.B)

            nengo.Connection(deconv.output, model.result.state.input,
                             transform=2)

            nengo.Connection(model.bound.state.output, model.bound.state.input,
                             synapse=self.mem_tau)

            vocab = model.get_output_vocab('result')
            vocab.parse('RED+BLUE+CIRCLE+SQUARE')
            model.cleanup = spa.AssociativeMemory(vocab)

            model.clean_result = spa.Buffer(self.dimensions)

            nengo.Connection(model.result.state.output, model.cleanup.input)
            nengo.Connection(model.cleanup.output,
                             model.clean_result.state.input)

            stim_time = self.mem_tau / self.mem_input_scale
            self.stim_time = stim_time

            def stim_color(t):
                if 0 < t < stim_time:
                    return 'BLUE'
                elif stim_time < t < stim_time * 2:
                    return 'RED'
                else:
                    return '0'

            def stim_shape(t):
                if 0 < t < stim_time:
                    return 'CIRCLE'
                elif stim_time < t < stim_time * 2:
                    return 'SQUARE'
                else:
                    return '0'

            def stim_query(t):
                if t < stim_time * 2:
                    return '0'
                else:
                    index = int((t - stim_time * 2) / self.test_present_time)
                    return ['BLUE', 'RED', 'CIRCLE', 'SQUARE'][index % 4]

            model.input = spa.Input(
                shape=stim_shape,
                color=stim_color,
                query=stim_query,
            )

            self.probe = nengo.Probe(model.clean_result.state.output,
                                     synapse=0.02)
            self.probe_wm = nengo.Probe(model.bound.state.output, synapse=0.02)

        self.vocab = model.get_output_vocab('clean_result')
        self.vocab_wm = model.get_output_vocab('bound')
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        stim_time = self.stim_time
        T = stim_time * 2 + self.test_time
        start = timeit.default_timer()
        sim.run(T, **kwargs)
        end = timeit.default_timer()
        speed = T / (end - start)

        vocab = self.vocab
        vals = [None] * 4
        vals[0] = np.dot(sim.data[self.probe], vocab.parse('CIRCLE').v)
        vals[1] = np.dot(sim.data[self.probe], vocab.parse('SQUARE').v)
        vals[2] = np.dot(sim.data[self.probe], vocab.parse('BLUE').v)
        vals[3] = np.dot(sim.data[self.probe], vocab.parse('RED').v)
        vals = np.array(vals)

        vocab_wm = self.vocab_wm
        vals_wm = [None] * 2
        vals_wm[0] = np.dot(sim.data[self.probe_wm],
                            vocab_wm.parse('BLUE*CIRCLE').v)
        vals_wm[1] = np.dot(sim.data[self.probe_wm],
                            vocab_wm.parse('RED*SQUARE').v)
        vals_wm = np.array(vals_wm)

        times = []
        recall_strength = []
        index = 0
        t = stim_time * 2 + self.test_present_time
        while t < T:
            i = int(t / sim.dt) - 1
            v = vals[index, i]
            recall_strength.append(v)
            index = (index + 1) % 4
            times.append(t)
            t += self.test_present_time

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), vals.T)
            plt.legend(['CIRCLE', 'SQUARE', 'BLUE', 'RED'], loc='best')
            for t in times:
                plt.axvline(t)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), vals_wm.T)

        return dict(mean_recall_strength=np.mean(recall_strength),
                    speed=speed)
