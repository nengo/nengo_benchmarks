"""
Nengo Benchmark Model #1: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

D = 2       # number of dimensions
L = 2       # number of layers
N = 100     # number of neurons per layer
pstc = 0.01 # synaptic time constant
T = 1.0     # amount of time to run for

import numpy as np

import nengo

model = nengo.Network()
with model:
    value = np.random.randn(D)
    value /= np.linalg.norm(value)

    input = nengo.Node(value)

    layers = [nengo.Ensemble(N, D) for i in range(L)]

    nengo.Connection(input, layers[0])
    for i in range(L-1):
        nengo.Connection(layers[i], layers[i+1], synapse=pstc)

    pInput = nengo.Probe(input)
    pOutput = nengo.Probe(layers[-1], synapse=pstc)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[pOutput])
pylab.plot(sim.trange(), sim.data[pInput])
pylab.show()

