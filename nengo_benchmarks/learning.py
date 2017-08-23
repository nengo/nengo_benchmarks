
import pytry

import nengo
import numpy as np
import timeit


class LearningSpeedup(pytry.NengoTrial):
    def params(self):
        self.param('dimensionality', D=1)
        self.param('number of neurons', n_neurons=100)
        self.param('slow path time constant', tau_slow=0.2)
        self.param('fast path time constant', tau_fast=0.01)
        self.param('time to simulate for', T=40.0)
        self.param('number of time to change function', n_switches=2)
        self.param('learning rate', learn_rate=1e-4)

    def model(self, p):
        model = nengo.Network()

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
                      learning_rule_type=nengo.PES(learning_rate=p.learn_rate))

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

            self.probe_target = nengo.Probe(target, synapse=p.tau_fast)
            self.probe_post = nengo.Probe(post, synapse=p.tau_fast)
            self.probe_pre = nengo.Probe(pre_value, synapse=None)
            self.probe_context = nengo.Probe(context, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        sim.run(p.T)
        end = timeit.default_timer()
        speed = p.T / (end - start)

        ideal = sim.data[self.probe_pre] * sim.data[self.probe_context]
        for i in range(2):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.tau_fast), p.dt)


        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.probe_target])
            plt.plot(sim.trange(), ideal)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), sim.data[self.probe_post])
            plt.plot(sim.trange(), ideal)

        rmse = np.sqrt(np.mean((sim.data[self.probe_post] - ideal)**2))
        return dict(rmse=rmse, speed=speed)
