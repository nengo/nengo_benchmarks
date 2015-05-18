"""
Nengo Benchmark Model #2: Lorenz Attractor

Input: none
Ouput: the 3 state variables for the classic Lorenz attractor
"""

N = 2000      # number of neurons
tau = 0.1     # post-synaptic time constant
sigma = 10    # Lorenz variables
beta = 8.0/3  # Lorenz variables
rho = 28      # Lorenz variables
T = 100.0     # time to run the simulation for

import nengo

model = nengo.Network()
with model:
    state = nengo.Ensemble(N, 3, radius=60)

    def feedback(x):
        dx0 = -sigma * x[0] + sigma * x[1]
        dx1 = -x[0] * x[2] - x[1]
        dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
        return [dx0 * tau + x[0],
                dx1 * tau + x[1],
                dx2 * tau + x[2]]
    nengo.Connection(state, state, function=feedback, synapse=tau)

    pState = nengo.Probe(state, 'decoded_output', synapse=tau)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[pState])
pylab.show()

