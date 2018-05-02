import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("matrix_mult")
class MatrixMultiply(object):
    """
    Nengo Benchmark Model: Matrix Multiplication

    Input: two random matrices of size D1xD2 and D2xD3
    Output: a D1xD3 matrix that is the product of the two inputs

    Parameters
    ----------
    n_neurons : int
        Number of neurons for multiplication
    n_neurons_io : int
        Number of neurons per dimension for input/output buffers
    d1 : int
        Size of matrices
    d2 : int
        Size of matrices
    d3 : int
        Size of matrices
    radius : floaat
        Range of values
    pstc : float
        Post-synaptic time constant
    sim_time : float
        Time to run simulation
    """

    def __init__(self, n_neurons=800, n_neurons_io=50, d1=1, d2=2, d3=2,
                 radius=1, pstc=0.01, sim_time=0.5):
        self.n_neurons = n_neurons
        self.n_neurons_io = n_neurons_io
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.radius = radius
        self.pstc = pstc
        self.sim_time = sim_time

    def model(self):
        model = nengo.Network()
        inputA = np.random.uniform(-self.radius, self.radius,
                                   self.d1 * self.d2)
        inputB = np.random.uniform(-self.radius, self.radius,
                                   self.d2 * self.d3)
        answer = np.dot(inputA.reshape(self.d1, self.d2),
                        inputB.reshape(self.d2, self.d3)).flatten()

        with model:
            inA = nengo.Node(inputA, label='inA')
            inB = nengo.Node(inputB, label='inB')
            ideal = nengo.Node(answer, label='ideal')

            A = nengo.networks.EnsembleArray(
                self.n_neurons_io, n_ensembles=self.d1 * self.d2,
                radius=self.radius, label='A')
            B = nengo.networks.EnsembleArray(
                self.n_neurons_io, n_ensembles=self.d2 * self.d3,
                radius=self.radius, label='B')
            D = nengo.networks.EnsembleArray(
                self.n_neurons_io, n_ensembles=self.d1 * self.d3,
                radius=self.radius, label='D')

            encoders = nengo.dists.Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

            # the C matrix holds the intermediate product calculations
            # need to compute D1*D2*D3 products to multiply 2 matrices together
            C = nengo.networks.EnsembleArray(
                self.n_neurons // (self.d1 * self.d2 * self.d3),
                n_ensembles=self.d1 * self.d2 * self.d3,
                label='C', radius=1.5 * self.radius, ens_dimensions=2,
                encoders=encoders)

            nengo.Connection(inA, A.input, synapse=self.pstc)
            nengo.Connection(inB, B.input, synapse=self.pstc)

            # determine the transformation matrices to get the correct pairwise
            # products computed.  This looks a bit like black magic but if
            # you manually try multiplying two matrices together, you can see
            # the underlying pattern.  Basically, we need to build up D1*D2*D3
            # pairs of numbers in C to compute the product of.  If i,j,k are
            # the indexes into the D1*D2*D3 products, we want to compute the
            # product of element (i,j) in A with the element (j,k) in B.  The
            # index in A of (i,j) is j+i*D2 and the index in B of (j,k) is
            # k+j*D3. The index in C is j+k*D2+i*D2*D3, multiplied by 2 since
            # there are two values per ensemble.  We add 1 to the B index so it
            # goes into the second value in the ensemble.
            transformA = [[0] * (self.d1 * self.d2) for _ in
                          range(self.d1 * self.d2 * self.d3 * 2)]
            transformB = [[0] * (self.d2 * self.d3) for _ in
                          range(self.d1 * self.d2 * self.d3 * 2)]
            for i in range(self.d1):
                for j in range(self.d2):
                    for k in range(self.d3):
                        transformA[
                            (j + k * self.d2 + i * self.d2 * self.d3) * 2][
                            j + i * self.d2] = 1
                        transformB[
                            (j + k * self.d2 + i * self.d2 * self.d3) * 2 + 1][
                            k + j * self.d3] = 1

            nengo.Connection(A.output, C.input, transform=transformA,
                             synapse=self.pstc)
            nengo.Connection(B.output, C.input, transform=transformB,
                             synapse=self.pstc)

            # now compute the products and do the appropriate summing
            def product(x):
                return x[0] * x[1]

            C.add_output('product', product)

            # the mapping for this transformation is much easier,
            # since we want to
            # combine D2 pairs of elements (we sum D2 products together)
            nengo.Connection(
                C.product, D.input[[i // self.d2 for i in
                                    range(self.d1 * self.d2 * self.d3)]],
                synapse=self.pstc)

            self.p_A = nengo.Probe(A.output, synapse=self.pstc)
            self.p_B = nengo.Probe(B.output, synapse=self.pstc)
            self.p_D = nengo.Probe(D.output, synapse=self.pstc)
            self.p_ideal = nengo.Probe(ideal, synapse=None)
        return model

    def evaluate(self, sim, plt=None, **kwargs):
        start = timeit.default_timer()
        sim.run(self.sim_time, **kwargs)
        end = timeit.default_timer()
        speed = self.sim_time / (end - start)

        ideal = sim.data[self.p_ideal]
        for i in range(4):
            ideal = nengo.Lowpass(self.pstc).filt(ideal, dt=sim.dt, y0=0)

        if plt is not None:
            plt.subplot(1, 3, 1)
            plt.plot(sim.trange(), sim.data[self.p_A])
            plt.ylim(-self.radius, self.radius)
            plt.subplot(1, 3, 2)
            plt.plot(sim.trange(), sim.data[self.p_B])
            plt.ylim(-self.radius, self.radius)
            plt.subplot(1, 3, 3)
            plt.plot(sim.trange(), sim.data[self.p_D])
            plt.plot(sim.trange(), ideal)
            plt.ylim(-self.radius, self.radius)

        rmse = np.sqrt(np.mean((sim.data[self.p_D] - ideal) ** 2))
        return dict(rmse=rmse, speed=speed)
