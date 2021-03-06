"""
Nengo Benchmark Model: SPA Sequence

Given no input, the model will cycle between cortical states using a
basal ganglia and thalamus.
"""


import pytry
import numpy as np
import nengo
import nengo.spa as spa
import timeit

class SPASequence(pytry.NengoTrial):
    def params(self):
        self.param('dimensionality', D=32)
        self.param('number of actions', n_actions=5)
        self.param('time to simulate', T=1.0)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.state = spa.Memory(dimensions=p.D)
            actions = ['dot(state, S%d) --> state=S%d' % (i,(i+1)%p.n_actions)
                       for i in range(p.n_actions)]
            model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
            model.thal = spa.Thalamus(model.bg)

            def state_input(t):
                if t < 0.1:
                    return 'S0'
                else:
                    return '0'
            model.input = spa.Input(state=state_input)

            self.probe = nengo.Probe(model.thal.actions.output, synapse=0.03)
        return model

    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        sim.run(p.T)
        end = timeit.default_timer()
        speed = p.T / (end - start)

        index = int(0.05 / p.dt)  # ignore the first 50ms
        best = np.argmax(sim.data[self.probe][index:], axis=1)
        change = np.diff(best)
        change_points = np.where(change != 0)[0]
        intervals = np.diff(change_points * p.dt)

        data = sim.data[self.probe][index:]
        peaks = [np.max(data[change_points[i]:change_points[i+1]])
                 for i in range(len(change_points)-1)]

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.plot(sim.trange()[index + 1:], np.where(change!=0,1,0))
            for i, peak in enumerate(peaks):
                plt.plot([p.dt*(change_points[i]+index), 
                          p.dt*(change_points[i+1]+index)],
                         [peak, peak], color='b')
            plt.xlabel('time (s)')
            plt.ylabel('action')
            plt.legend(['action %d' % i for i in range(p.n_actions)], loc='best')

        return dict(period=np.mean(intervals), period_sd=np.std(intervals),
                    peak=np.mean(peaks), peak_sd=np.std(peaks),
                    speed=speed)
