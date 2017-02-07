"""Controlled Oscillator Benchmark

This benchmark builds a controlled oscillator and tests it at a sequence of
inputs. The benchmark computes the FFT of the response of the system and
compares that to the FFT of perfect sine waves of the desired frequencies.
The final score is the mean normalized dot product between the FFTs.
"""

import nengo
import numpy as np
import pytry

class Oscillator(pytry.NengoTrial):
    def params(self):
        self.param('maximum frequency', f_max=2.0)
        self.param('time per freq', T=10.0)
        self.param('number of freqs', n_freq=7)
        self.param('number of oscillator neurons', N_osc=1000)
        self.param('number of control neurons', N_control=100)
        self.param('post-synaptic time constant', pstc=0.1)

    def model(self, p):
        self.stims = np.linspace(-1, 1, p.n_freq)

        model = nengo.Network()
        with model:
            state = nengo.Ensemble(n_neurons=p.N_osc, dimensions=3,
                                   radius=1.7)

            def feedback(x):
                x0, x1, f = x
                w = f * p.f_max * 2 * np.pi
                return x0 + w * p.pstc * x1, x1 - w * p.pstc * x0
            nengo.Connection(state, state[:2],
                             function=feedback, synapse=p.pstc)

            freq = nengo.Ensemble(n_neurons=p.N_control, dimensions=1)
            nengo.Connection(freq, state[2], synapse=p.pstc)

            stim = nengo.Node(lambda t: [1,0,0] if t < 0.08 else [0,0,0])
            nengo.Connection(stim, state)

            def control(t):
                index = int(t / p.T) % p.n_freq
                return self.stims[index]
            freq_control = nengo.Node(control)

            nengo.Connection(freq_control, freq)

            self.p_state = nengo.Probe(state, synapse=0.03)
            self.p_freq = nengo.Probe(freq, synapse=0.03)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T * p.n_freq)

        data = sim.data[self.p_state][:, 1]


        ideal_freqs = p.f_max * self.stims # target frequencies

        steps = int(p.T / p.dt)
        freqs = np.fft.fftfreq(steps, d=p.dt)

        # compute fft for each input
        data.shape = p.n_freq, steps
        fft = np.fft.fft(data, axis=1)

        # compute ideal fft for each input
        ideal_data = np.zeros_like(data)
        for i in range(p.n_freq):
            ideal_data[i] = np.cos(2 * np.pi * ideal_freqs[i] *
                                   np.arange(steps) * p.dt)
        ideal_fft = np.fft.fft(ideal_data, axis=1)

        # only consider the magnitude
        fft = np.abs(fft)
        ideal_fft = np.abs(ideal_fft)

        # compute the normalized dot product between the actual and ideal ffts
        score = np.zeros(p.n_freq)
        for i in range(p.n_freq):
            score[i] = np.dot(fft[i] / np.linalg.norm(fft[i]),
                              ideal_fft[i] / np.linalg.norm(ideal_fft[i]))

        if plt is not None:
            plt.subplot(2, 1, 1)
            lines = plt.plot(np.fft.fftshift(freqs),
                             np.fft.fftshift(fft, axes=1).T)
            plt.xlim(-p.f_max * 2, p.f_max * 2)
            plt.xlabel('FFT of decoded value (Hz)')
            plt.title('Score: %1.4f' % np.mean(score))
            plt.legend(lines, ['%1.3f' % s for s in score],
                         loc='best', prop={'size': 8}, title='score')

            plt.subplot(2, 1, 2)
            lines = plt.plot(np.arange(steps) * p.dt, data.T)
            plt.xlabel('decoded value')
            plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
                         loc='best', prop={'size': 8})

        return dict(scores=score,
                    mean_score=np.mean(score))
