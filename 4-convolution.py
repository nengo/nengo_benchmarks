"""
Nengo Benchmark Model #4: Circular Convolution

Input: two random D-dimensional vectors
Output: the circular convolution of the inputs

"""

D = 16       # dimensionality
T = 0.5      # time to run the simulation for

import nengo
import nengo.spa as spa

model = spa.SPA()
with model:
    model.inA = spa.Buffer(D)
    model.inB = spa.Buffer(D)

    model.result = spa.Buffer(D)

    model.cortical = spa.Cortical(spa.Actions('result = inA * inB'))

    model.input = spa.Input(inA='A', inB='B')

    probe = nengo.Probe(model.result.state.output, synapse=0.05)

    ideal = nengo.Node(model.get_output_vocab('inA').parse('A*B').v)
    probe_ideal = nengo.Probe(ideal)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[probe])
pylab.plot(sim.trange(), sim.data[probe_ideal])
pylab.show()

