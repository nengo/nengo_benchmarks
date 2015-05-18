"""
Nengo Benchmark Model #6: Simple Parsing

The model parses and executes simple commands sequentially presented to it

"""

dimensions = 32  # dimensionality of the cortical representation
T = 1.5          # time to run the simulation for

import numpy as np

import nengo
import nengo.spa as spa

model = spa.SPA()
with model:
    model.vision = spa.Buffer(dimensions=dimensions)
    model.phrase = spa.Buffer(dimensions=dimensions)
    model.motor = spa.Buffer(dimensions=dimensions)
    model.noun = spa.Memory(dimensions=dimensions)
    model.verb = spa.Memory(dimensions=dimensions)

    model.bg = spa.BasalGanglia(spa.Actions(
        'dot(vision, WRITE) --> verb=vision',
        'dot(vision, ONE+TWO+THREE) --> noun=vision',
        '0.5*(dot(vision, NONE-WRITE-ONE-TWO-THREE) + dot(phrase, WRITE*VERB))'
             '--> motor=phrase*~NOUN',
        ))
    model.thal = spa.Thalamus(model.bg)

    model.cortical = spa.Cortical(spa.Actions(
        'phrase=noun*NOUN',
        'phrase=verb*VERB',
        ))

    def vision_input(t):
        t = t % 1.5
        if t<0.5: return 'WRITE'
        elif t<1.0: return 'ONE'
        elif t<1.5: return 'NONE'
        else: return '0'
    model.input = spa.Input(vision=vision_input)

    p = nengo.Probe(model.thal.actions.output, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[p])
pylab.show()
