import logging

import nengo
from nengo.utils.numpy import maxseed, rms
import numpy as np
import pytest


class HilbertCurve:
    """Hilbert curve function.

    Pre-calculates the Hilbert space filling curve with a given number
    of iterations. The curve will lie in the square delimited by the
    points (0, 0) and (1, 1).

    Arguments
    ---------
    n : int
        Iterations.
    """

    # Implementation based on
    # https://en.wikipedia.org/w/index.php?title=Hilbert_curve&oldid=633637210

    def __init__(self, n):
        self.n = n
        self.n_corners = (2 ** n) ** 2
        self.corners = np.zeros((self.n_corners, 2))
        self.steps = np.arange(self.n_corners)

        steps = np.arange(self.n_corners)
        for s in 2 ** np.arange(n):
            r = np.empty_like(self.corners, dtype="int")
            r[:, 0] = 1 & (steps // 2)
            r[:, 1] = 1 & (steps ^ r[:, 0])
            self._rot(s, r)
            self.corners += s * r
            steps //= 4

        self.corners /= (2 ** n) - 1

    def _rot(self, s, r):
        swap = r[:, 1] == 0
        flip = np.all(r == np.array([1, 0]), axis=1)

        self.corners[flip] = s - 1 - self.corners[flip]
        self.corners[swap] = self.corners[swap, ::-1]

    def __call__(self, u):
        """Evaluate pre-calculated Hilbert curve.

        Arguments
        ---------
        u : ndarray (M,)
            Positions to evaluate on the curve in the range [0, 1].

        Returns
        -------
        ndarray (M, 2)
            Two-dimensional curve coordinates.
        """
        step = np.asarray(u * len(self.steps))
        return np.vstack(
            (
                np.interp(step, self.steps, self.corners[:, 0]),
                np.interp(step, self.steps, self.corners[:, 1]),
            )
        ).T


@pytest.mark.benchmark
@pytest.mark.slow
def test_product_benchmark(Simulator, analytics, rng):
    n_trials = 50
    hc = HilbertCurve(n=4)  # Increase n to cover the input space more densely
    duration = 5.0  # Simulation duration (s)
    # Duration (s) to wait at the beginning to have a stable representation
    wait_duration = 0.5
    n_neurons = 100
    n_eval_points = 1000

    def stimulus_fn(t):
        return np.squeeze(hc(t / duration).T * 2 - 1)

    def run_trial():
        model = nengo.Network(seed=rng.randint(maxseed))
        with model:
            model.config[nengo.Ensemble].n_eval_points = n_eval_points

            stimulus = nengo.Node(
                output=lambda t: stimulus_fn(max(0.0, t - wait_duration)), size_out=2
            )

            product_net = nengo.networks.Product(n_neurons, 1)
            nengo.Connection(stimulus[0], product_net.input_a)
            nengo.Connection(stimulus[1], product_net.input_b)
            probe_test = nengo.Probe(product_net.output)

            ens_direct = nengo.Ensemble(1, dimensions=2, neuron_type=nengo.Direct())
            result_direct = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens_direct)
            nengo.Connection(
                ens_direct, result_direct, function=lambda x: x[0] * x[1], synapse=None
            )
            probe_direct = nengo.Probe(result_direct)

        with Simulator(model) as sim:
            sim.run(duration + wait_duration, progress_bar=False)

        selection = sim.trange() > wait_duration
        test = sim.data[probe_test][selection]
        direct = sim.data[probe_direct][selection]
        return rms(test - direct)

    error_data = [run_trial() for i in range(n_trials)]
    analytics.add_data("error", error_data, "Multiplication RMSE. Shape: n_trials")


@pytest.mark.compare
def test_compare_product_benchmark(analytics_data):
    stats = pytest.importorskip("scipy.stats")
    data1, data2 = (d["error"] for d in analytics_data)
    improvement = np.mean(data1) - np.mean(data2)
    p = (
        np.ceil(
            1000.0 * 2.0 * stats.mannwhitneyu(data1, data2, alternative="two-sided")[1]
        )
        / 1000.0
    )
    logging.info(
        "Multiplication improvement by %f (%.0f%%, p < %.3f)",
        improvement,
        (1.0 - np.mean(data2) / np.mean(data1)) * 100.0,
        p,
    )
    assert improvement >= 0.0 or p >= 0.05
