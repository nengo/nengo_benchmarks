import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("oscillator")
class Oscillator(object):
    """
    Controlled Oscillator Benchmark

    This benchmark builds a controlled oscillator and tests it at a sequence of
    inputs. The benchmark computes the FFT of the response of the system and
    compares that to the FFT of perfect sine waves of the desired frequencies.
    The final score is the mean normalized dot product between the FFTs.
    
    Parameters
    ----------
    n_neurons : int
        Number of oscillator neurons
    n_neurons_ctrl : int
        Number of control neurons
    f_max : float
        Maximum frequency
    sim_time : float
        Simulation time per input
    n_freq : int
        Number of inputs
    pstc : float
        Post-synaptic time constant
    """

    def __init__(self, n_neurons=1000, n_neurons_ctrl=100, f_max=2.0,
                 sim_time=10.0, n_freq=7, pstc=0.1):
        self.n_neurons = n_neurons
        self.n_neurons_ctrl = n_neurons_ctrl
        self.f_max = f_max
        self.sim_time = sim_time
        self.n_freq = n_freq
        self.pstc = pstc

    def model(self):
        self.stims = np.linspace(-1, 1, self.n_freq)

        model = nengo.Network()
        with model:
            state = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=3,
                                   radius=1.7)

            def feedback(x):
                x0, x1, f = x
                w = f * self.f_max * 2 * np.pi
                return x0 + w * self.pstc * x1, x1 - w * self.pstc * x0

            nengo.Connection(state, state[:2],
                             function=feedback, synapse=self.pstc)

            freq = nengo.Ensemble(n_neurons=self.n_neurons_ctrl, dimensions=1)
            nengo.Connection(freq, state[2], synapse=self.pstc)

            stim = nengo.Node(lambda t: [1, 0, 0] if t < 0.08 else [0, 0, 0])
            nengo.Connection(stim, state)

            def control(t):
                index = int(t / self.sim_time) % self.n_freq
                return self.stims[index]

            freq_control = nengo.Node(control)

            nengo.Connection(freq_control, freq)

            self.p_state = nengo.Probe(state, synapse=0.03)
            self.p_freq = nengo.Probe(freq, synapse=0.03)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        T = self.sim_time * self.n_freq
        start = timeit.default_timer()
        sim.run(T, **kwargs)
        end = timeit.default_timer()
        speed = T / (end - start)

        data = sim.data[self.p_state][:, 1]

        ideal_freqs = self.f_max * self.stims  # target frequencies

        steps = int(self.sim_time / sim.dt)
        freqs = np.fft.fftfreq(steps, d=sim.dt)

        # compute fft for each input
        data.shape = self.n_freq, steps
        fft = np.fft.fft(data, axis=1)

        # compute ideal fft for each input
        ideal_data = np.zeros_like(data)
        for i in range(self.n_freq):
            ideal_data[i] = np.cos(2 * np.pi * ideal_freqs[i] *
                                   np.arange(steps) * sim.dt)
        ideal_fft = np.fft.fft(ideal_data, axis=1)

        # only consider the magnitude
        fft = np.abs(fft)
        ideal_fft = np.abs(ideal_fft)

        # compute the normalized dot product between the actual and ideal ffts
        score = np.zeros(self.n_freq)
        for i in range(self.n_freq):
            score[i] = np.dot(fft[i] / np.linalg.norm(fft[i]),
                              ideal_fft[i] / np.linalg.norm(ideal_fft[i]))

        if plt is not None:
            plt.subplot(2, 1, 1)
            lines = plt.plot(np.fft.fftshift(freqs),
                             np.fft.fftshift(fft, axes=1).T)
            plt.xlim(-self.f_max * 2, self.f_max * 2)
            plt.xlabel('FFT of decoded value (Hz)')
            plt.title('Score: %1.4f' % np.mean(score))
            plt.legend(lines, ['%1.3f' % s for s in score],
                       loc='best', prop={'size': 8}, title='score')

            plt.subplot(2, 1, 2)
            lines = plt.plot(np.arange(steps) * sim.dt, data.T)
            plt.xlabel('decoded value')
            plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
                       loc='best', prop={'size': 8})

        return dict(scores=score,
                    mean_score=np.mean(score),
                    speed=speed)
