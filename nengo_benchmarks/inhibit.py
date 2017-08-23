import nengo
import pytry
import numpy as np
import timeit

class InhibitionTrial(pytry.NengoTrial):
    def params(self):
        self.param('number of neurons', n_neurons=100)
        self.param('number of dimensions', D=1)
        self.param('inhibition strength', inh_strength=2.0)
        self.param('time to run', T=3.0)
        
    def model(self, p):
        model = nengo.Network()
        with model:
            stim = nengo.Node(1)
            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D)
            nengo.Connection(stim, ens, transform=np.ones((p.D, 1)),
                             synapse=None)

            stim_inh = nengo.Node(lambda t: t)
            inh = nengo.Ensemble(n_neurons=100, dimensions=1)
            nengo.Connection(stim_inh, inh, synapse=None)
            nengo.Connection(inh, ens.neurons, 
                    transform=np.ones((p.n_neurons, 1))*(-p.inh_strength),
                    synapse=None)

            self.p_ens = nengo.Probe(ens, synapse=0.01)
        return model

    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        sim.run(p.T)
        end = timeit.default_timer()
        speed = p.T / (end - start)

        data = sim.data[self.p_ens]

        last = []
        for row in data.T:
            nz = np.nonzero(row>0.05)[0]
            if len(nz) == 0:
                last.append(0)
            else:
                last.append(nz[-1])
        time_to_inhibit = np.array(last)*p.dt

        if plt:
            plt.plot(sim.trange(), sim.data[self.p_ens])
            for t in time_to_inhibit:
                plt.axvline(t)
            plt.axhline(0.05, linestyle='--', c='k')

        return dict(time_to_inhibit=np.mean(time_to_inhibit),
                    speed=speed)
