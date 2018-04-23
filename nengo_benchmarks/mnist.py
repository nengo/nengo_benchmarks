import nengo
import numpy as np
import timeit

import nengo_benchmarks


@nengo_benchmarks.register("mnist")
class MNIST(object):
    """
    Nengo Benchmark Model: MNIST

    Input: Images from the MNIST data set
    Ouput: the categorization of the image

    Parameters
    ----------
    n_neurons : int
        Number of neurons in hidden layer
    t_image : float
        Time per image
    n_training : int
        Number of training samples
    n_testing : int
        Number of testing samples
    synapse : float
        Output synapse
    use_gabor : bool
        If True, use Gabor encoders
    n_backprop : int
        Iterations of backprop to run
    learning_rate : float
        Learning rate
    n_feedback : int
        Iterations of feedback alignment to run
    reg : float
        L2 regularization in decoder solving
    """

    def __init__(self, n_neurons=500, t_image=0.1, n_training=5000,
                 n_testing=100, synapse=0.02, use_gabor=False, n_backprop=0,
                 learning_rate=1e-3, n_feedback=0, reg=0.1):
        self.n_neurons = n_neurons
        self.t_image = t_image
        self.n_training = n_training
        self.n_testing = n_testing
        self.synapse = synapse
        self.use_gabor = use_gabor
        self.n_backprop = n_backprop
        self.learning_rate = learning_rate
        self.n_feedback = n_feedback
        self.reg = reg

    def model(self):
        import sklearn.datasets
        mnist = sklearn.datasets.fetch_mldata('MNIST original')

        x = mnist['data'].astype(float) - 128
        x = x / np.linalg.norm(x, axis=1)[:, None]
        y = mnist['target']
        y = np.eye(10)[y.astype(int)] * 2 - 1
        y = y / np.linalg.norm(y, axis=1)[:, None]

        order = np.arange(len(x))
        np.random.shuffle(order)
        x = x[order]
        y = y[order]

        model = nengo.Network()
        with model:
            def stim_func(t, data=x[self.n_training:]):
                index = int(t / self.t_image)
                return data[index % len(data)]

            stim = nengo.Node(stim_func)

            if self.use_gabor:
                from nengo_extras.vision import Gabor, Mask
                encoders = Gabor().generate(self.n_neurons, (11, 11))
                encoders = Mask((28, 28)).populate(encoders, flatten=True)
            else:
                encoders = nengo.dists.UniformHypersphere(surface=True)

            ens = nengo.Ensemble(
                n_neurons=self.n_neurons, dimensions=784,
                encoders=encoders,
                intercepts=nengo.dists.CosineSimilarity(784 + 2))

            def result_func(t, raw, correct=y[self.n_training:]):
                index = int(t / self.t_image) - 1
                c = np.argmax(correct[index])
                r = np.argmax(raw)
                return np.hstack([raw, c, r])

            result = nengo.Node(result_func, size_in=10)

            nengo.Connection(stim, ens, synapse=None)
            c = nengo.Connection(
                ens, result, eval_points=x[:self.n_training],
                function=y[:self.n_training],
                solver=nengo.solvers.LstsqL2(reg=self.reg),
                synapse=self.synapse)

            self.p_output = nengo.Probe(result)

        if self.n_backprop > 0:
            import nengo_encoder_learning as nel
            nel.improve(c, backprop=True,
                        learning_rate=self.learning_rate,
                        steps=self.n_backprop,
                        seed=np.random.randint(0x7FFFFFFF))
        if self.n_feedback > 0:
            import nengo_encoder_learning as nel
            nel.improve(c, backprop=False,
                        learning_rate=self.learning_rate,
                        steps=self.n_feedback,
                        seed=np.random.randint(0x7FFFFFFF))
        return model

    def evaluate(self, sim, plt=None):
        start = timeit.default_timer()
        T = self.n_testing * self.t_image
        sim.run(T)
        end = timeit.default_timer()
        speed = T / (end - start)

        correct = sim.data[self.p_output][:, 10].astype(int)
        result = sim.data[self.p_output][:, 11].astype(int)
        confusion = np.zeros((10, 10), dtype=int)
        count = np.zeros(10, dtype=int)
        times = sim.trange()
        for i in range(self.n_testing):
            t = (i + 1) * self.t_image
            index = np.argmax(times >= t) - 1
            count[correct[index]] += 1
            confusion[correct[index], result[index]] += 1

        score = sum(confusion[i, i] for i in range(10)) / float(self.n_testing)

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.p_output][:, :10])
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), sim.data[self.p_output][:, 10:])

        return dict(speed=speed, score=score, count=count, confusion=confusion)
