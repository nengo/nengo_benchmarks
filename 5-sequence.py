"""
Nengo Benchmark Model #5: SPA Sequence

Given no input, the model will cycle between cortical states using a
basal ganglia and thalamus.

"""


dimensions = 32  # dimensionality of the cortical representation
action_count = 5 # number of states to transition between
T = 1.0          # time to run the simulation for

import numpy as np

import nengo
import nengo.spa as spa

model = spa.SPA()
with model:
    model.state = spa.Memory(dimensions=dimensions)
    actions = ['dot(state, S%d) --> state=S%d' % (i, (i+1)%action_count)
               for i in range(action_count)]
    model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
    model.thal = spa.Thalamus(model.bg)

    def state_input(t):
        if t<0.1: return 'S0'
        else: return '0'
    model.input = spa.Input(state=state_input)

    p = nengo.Probe(model.thal.actions.output, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[p])
pylab.show()
