
import benchmark

import nengo
import numpy as np
import time


class LearningSpeedup(benchmark.Benchmark):
    def params(self):
        return dict(
            D=1,
            n_neurons=100,
            tau_slow=0.2,
            tau_fast=0.01,
            T=40.0,
            n_switches=2,
            learn_rate=1.0,
        )
    def benchmark(self, p, Simulator, rng, plt):
        model = nengo.Network(seed=p.seed)

        with model:
            def stim(t):
                return [np.sin(t+i*np.pi*2/p.D) for i in range(p.D)]
            pre_value = nengo.Node(stim)

            pre = nengo.Ensemble(p.n_neurons, p.D)
            post = nengo.Ensemble(p.n_neurons, p.D)
            target = nengo.Ensemble(p.n_neurons, p.D)
            nengo.Connection(pre_value, pre, synapse=None)

            conn = nengo.Connection(pre, post,
                        function=lambda x: np.random.random(size=p.D),
                        learning_rule_type=nengo.PES())
            conn.learning_rule_type.learning_rate *= p.learn_rate

            slow = nengo.networks.Product(p.n_neurons*2, p.D)
            T_context = p.T / p.n_switches
            context = nengo.Node(lambda t: 1 if int(t/T_context)%2 else -1)

            nengo.Connection(context, slow.A, transform=np.ones((p.D,1)))

            nengo.Connection(pre, slow.B, synapse=p.tau_slow)

            nengo.Connection(slow.output, target, synapse=p.tau_slow)

            error = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)

            nengo.Connection(post, error, synapse=p.tau_slow*2+p.tau_fast)
            nengo.Connection(target, error, transform=-1, synapse=p.tau_fast)

            nengo.Connection(error, conn.learning_rule)

            probe_target = nengo.Probe(target, synapse=p.tau_fast)
            probe_post = nengo.Probe(post, synapse=p.tau_fast)
            probe_pre = nengo.Probe(pre_value, synapse=None)
            probe_context = nengo.Probe(context, synapse=None)

        time_start = time.time()
        sim = Simulator(model, dt=p.dt)
        time_built = time.time()
        sim.run(p.T)
        time_ran = time.time()

        ideal = sim.data[probe_pre] * sim.data[probe_context]
        for i in range(2):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.tau_fast), p.dt)


        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_target])
            plt.plot(sim.trange(), ideal)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), sim.data[probe_post])
            plt.plot(sim.trange(), ideal)

        rmse = np.sqrt(np.mean(sim.data[probe_post] - ideal)**2)
        return dict(rmse=rmse,
                    time_build=time_built - time_start,
                    rate=p.T / (time_ran - time_built))


if __name__ == '__main__':
    b = LearningSpeedup().run()
