"""
Nengo Benchmark Model: MNIST

Input: Images from the MNIST data set
Ouput: the categorization of the image
"""

import nengo
import numpy as np
import pytry
import timeit
import sklearn.datasets

class MNIST(pytry.NengoTrial):
    def params(self):
        self.param('number of neurons', n_neurons=500)
        self.param('time per image', t_image=0.1)
        self.param('training samples', n_training=5000)
        self.param('testing samples', n_testing=100)
        self.param('output synapse', synapse=0.02)

    def model(self, p):
        mnist = sklearn.datasets.fetch_mldata('MNIST original')

        x = mnist['data'].astype(float)-128
        x = x/np.linalg.norm(x, axis=1)[:,None]
        y = mnist['target']
        y = np.eye(10)[y.astype(int)]*2-1
        y = y/np.linalg.norm(y, axis=1)[:,None]

        order = np.arange(len(x))
        np.random.shuffle(order)
        x = x[order]
        y = y[order]
        
        model = nengo.Network()
        with model:
            def stim_func(t, data=x[p.n_training:]):
                index = int(t / p.t_image)
                return data[index % len(data)]
            stim = nengo.Node(stim_func)

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=784,
                                 intercepts=nengo.dists.CosineSimilarity(784+2))
            
            def result_func(t, raw, correct=y[p.n_training:]):
                index = int(t / p.t_image)
                c = np.argmax(correct[index])
                r = np.argmax(raw)
                return np.hstack([raw, c, r])
            result = nengo.Node(result_func, size_in=10)

            nengo.Connection(stim, ens, synapse=None)
            nengo.Connection(ens, result, 
                             eval_points=x[:p.n_training], 
                             function=y[:p.n_training],
                             synapse=p.synapse)

            self.p_output = nengo.Probe(result)
        return model


    def evaluate(self, p, sim, plt):
        start = timeit.default_timer()
        T = p.n_testing * p.t_image
        sim.run(T)
        end = timeit.default_timer()
        speed = T / (end - start)

        correct = sim.data[self.p_output][:,10].astype(int)
        result = sim.data[self.p_output][:,11].astype(int)
        confusion = np.zeros((10,10), dtype=int)
        count = np.zeros(10, dtype=int)
        times = sim.trange()
        for i in range(p.n_testing):
            t = (i + 0.9) * p.t_image
            index = np.argmax(times >= t)
            count[correct[index]] += 1
            confusion[correct[index],result[index]] += 1

        score = sum(confusion[i, i] for i in range(10)) / float(p.n_testing)

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.p_output][:,:10])
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), sim.data[self.p_output][:,10:])

        return dict(speed=speed, score=score, count=count, confusion=confusion)
