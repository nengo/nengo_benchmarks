"""
Nengo Benchmark Model: Circular Convolution

Input: two random D-dimensional vectors
Output: the circular convolution of the inputs

"""

import pytry
import nengo
import nengo.spa as spa
import numpy as np
import timeit

class CircularConvolution(pytry.NengoTrial):
    def params(self):
        self.param('dimensionality', D=8)
        self.param('time to run simulation', T=0.5)
        self.param('subdimensions', SD=8)
        self.param('post-synaptic time constant', pstc=0.01)
        self.param('neurons per dimension I/O', n_per_d=50)
        self.param('neurons per cconv', n_cconv=200)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.inA = spa.Buffer(p.D, subdimensions=p.SD,
                                   neurons_per_dimension=p.n_per_d)
            model.inB = spa.Buffer(p.D, subdimensions=p.SD,
                                   neurons_per_dimension=p.n_per_d)

            model.result = spa.Buffer(p.D, subdimensions=p.SD,
                                      neurons_per_dimension=p.n_per_d)

            model.cortical = spa.Cortical(spa.Actions('result = inA * inB'),
                                          synapse=p.pstc,
                                          neurons_cconv=p.n_cconv)

            model.input = spa.Input(inA='A', inB='B')

            self.probe = nengo.Probe(model.result.state.output, synapse=p.pstc)

            ideal = nengo.Node(model.get_output_vocab('inA').parse('A*B').v)
            self.probe_ideal = nengo.Probe(ideal, synapse=None)

        return model

    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        sim.run(p.T)
        end = timeit.default_timer()
        speed = p.T / (end - start)

        ideal = sim.data[self.probe_ideal]
        for i in range(3):
            ideal = nengo.Lowpass(p.pstc).filt(ideal, dt=p.dt, y0=0)

        # compute where to check results from
        index = int(p.pstc*3*4 / p.dt)

        if plt is not None:
            for i in range(p.D):
                plt.plot(sim.trange(), sim.data[self.probe][:,i],
                         label='output (dim %d)'%i if i==0 else None)
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.gca().set_prop_cycle(None)
            for i in range(p.D):
                plt.plot(sim.trange(), ideal[:,i], ls='--',
                         label='ideal (dim %d)'%i if i==0 else None)
            plt.axvline(index*p.dt, c='#aaaaaa')
            plt.legend(loc='best')
            plt.xlabel('time (s)')
            plt.ylabel('decoded value')

        rmse = np.sqrt(np.mean((sim.data[self.probe][index:]-ideal[index:])**2))
        return dict(rmse=rmse, speed=speed)
