"""
Nengo Benchmark Model #3: Matrix Multiplication

Input: two random matrices of size D1xD2 and D2xD3
Output: a D1xD3 matrix that is the product of the two inputs

"""

D1 = 1       # size of matrices
D2 = 2       # size of matrices
D3 = 2       # size of matrices
radius = 1   # All values must be between -radius and radius
N = 50       # number of neurons per input and output value
N_mult = 200 # number of neurons to compute a pairwise product
pstc = 0.01  # post-synaptic time constant
T = 0.5      # time to run the simulation for

import nengo
import numpy as np
inputA = np.random.uniform(-radius, radius, D1*D2)
inputB = np.random.uniform(-radius, radius, D2*D3)
answer = np.dot(inputA.reshape(D1, D2), inputB.reshape(D2, D3)).flatten()

model = nengo.Network()

with model:
    inA = nengo.Node(inputA)
    inB = nengo.Node(inputB)
    ideal = nengo.Node(answer)

    A = nengo.networks.EnsembleArray(N, n_ensembles=D1*D2, radius=radius)
    B = nengo.networks.EnsembleArray(N, n_ensembles=D2*D3, radius=radius)
    D = nengo.networks.EnsembleArray(N, n_ensembles=D1*D3, radius=radius)

    encoders = nengo.dists.Choice([[1,1],[1,-1],[-1,1],[-1,-1]])

    # the C matrix holds the intermediate product calculations
    #  need to compute D1*D2*D3 products to multiply 2 matrices together
    C = nengo.networks.EnsembleArray(N_mult, n_ensembles=D1*D2*D3,
            radius=1.5*radius, ens_dimensions=2, encoders=encoders)

    nengo.Connection(inA, A.input, synapse=pstc)
    nengo.Connection(inB, B.input, synapse=pstc)

    #  determine the transformation matrices to get the correct pairwise
    #  products computed.  This looks a bit like black magic but if
    #  you manually try multiplying two matrices together, you can see
    #  the underlying pattern.  Basically, we need to build up D1*D2*D3
    #  pairs of numbers in C to compute the product of.  If i,j,k are the
    #  indexes into the D1*D2*D3 products, we want to compute the product
    #  of element (i,j) in A with the element (j,k) in B.  The index in
    #  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
    #  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
    #  two values per ensemble.  We add 1 to the B index so it goes into
    #  the second value in the ensemble.
    transformA = [[0]*(D1*D2) for i in range(D1*D2*D3*2)]
    transformB = [[0]*(D2*D3) for i in range(D1*D2*D3*2)]
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                transformA[(j + k*D2 + i*D2*D3)*2][j + i*D2] = 1
                transformB[(j + k*D2 + i*D2*D3)*2 + 1][k + j*D3] = 1

    nengo.Connection(A.output, C.input, transform=transformA, synapse=pstc)
    nengo.Connection(B.output, C.input, transform=transformB, synapse=pstc)


    # now compute the products and do the appropriate summing
    def product(x):
        return x[0]*x[1]

    C.add_output('product', product)

    # the mapping for this transformation is much easier, since we want to
    # combine D2 pairs of elements (we sum D2 products together)
    nengo.Connection(C.product, D.input[[i/D2 for i in range(D1*D2*D3)]], synapse=pstc)


    pA = nengo.Probe(A.output, synapse=pstc)
    pB = nengo.Probe(B.output, synapse=pstc)
    pD = nengo.Probe(D.output, synapse=pstc)
    pIdeal = nengo.Probe(ideal, synapse=pstc)

sim = nengo.Simulator(model)
sim.run(T)

import pylab
pylab.subplot(1,3,1)
pylab.plot(sim.trange(), sim.data[pA])
pylab.ylim(-radius, radius)
pylab.subplot(1,3,2)
pylab.plot(sim.trange(), sim.data[pB])
pylab.ylim(-radius, radius)
pylab.subplot(1,3,3)
pylab.plot(sim.trange(), sim.data[pD])
pylab.plot(sim.trange(), sim.data[pIdeal])
pylab.ylim(-radius, radius)
pylab.show()


