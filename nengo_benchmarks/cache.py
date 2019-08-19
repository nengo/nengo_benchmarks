import logging
import timeit
from textwrap import dedent

import numpy as np
import pytest


def reject_outliers(data):
    med = np.median(data)
    limits = 1.5 * (np.percentile(data, [25, 75]) - med) + med
    return np.asarray(data)[np.logical_and(data > limits[0], data < limits[1])]


class TestCacheBenchmark:
    n_trials = 25

    setup = dedent(
        """\
        import numpy as np
        import nengo
        import nengo.cache
        from nengo.rc import rc

        rc.set("decoder_cache", "path", {tmpdir!r})

        model = nengo.Network(seed=1)
        with model:
            a = nengo.Ensemble({N}, dimensions={D}, n_eval_points={M})
            b = nengo.Ensemble({N}, dimensions={D}, n_eval_points={M})
            conn = nengo.Connection(a, b)
        """
    )

    without_cache = {
        "rc": dedent(
            """\
            rc.set("progress", "progress_bar", "none")
            rc.set("decoder_cache", "enabled", "False")
            """
        ),
        "stmt": dedent(
            """\
            with nengo.Simulator(model):
                pass
            """
        ),
    }

    with_cache_miss_ro = {
        "rc": dedent(
            """\
            rc.set("progress", "progress_bar", "none")
            with nengo.cache.DecoderCache() as cache:
                cache.invalidate()
            rc.set("decoder_cache", "enabled", "True")
            rc.set("decoder_cache", "readonly", "True")
            """
        ),
        "stmt": dedent(
            """\
            with nengo.Simulator(model):
                pass
            """
        ),
    }

    with_cache_miss = {
        "rc": dedent(
            """\
            rc.set("progress", "progress_bar", "none")
            with nengo.cache.DecoderCache() as cache:
                cache.invalidate()
            rc.set("decoder_cache", "enabled", "True")
            rc.set("decoder_cache", "readonly", "False")
            """
        ),
        "stmt": dedent(
            """\
            with nengo.Simulator(model):
                pass
            """
        ),
    }

    with_cache_hit = {
        "rc": dedent(
            """\
            rc.set("progress", "progress_bar", "none")
            rc.set("decoder_cache", "enabled", "True")
            rc.set("decoder_cache", "readonly", "False")
            with nengo.Simulator(model):
                pass
            """
        ),
        "stmt": dedent(
            """\
            with nengo.Simulator(model):
                pass
            """
        ),
    }

    labels = ["no cache", "cache miss", "cache miss ro", "cache hit"]
    keys = [l.replace(" ", "_") for l in labels]
    param_to_axis_label = {"D": "dimensions", "N": "neurons", "M": "evaluation points"}
    defaults = {"D": 1, "N": 50, "M": 1000}

    def time_code(self, code, **kwargs):
        return timeit.repeat(
            stmt=code["stmt"],
            setup=self.setup.format(**kwargs) + code["rc"],
            number=1,
            repeat=self.n_trials,
        )

    def time_all(self, **kwargs):
        return (
            self.time_code(self.without_cache, **kwargs),
            self.time_code(self.with_cache_miss, **kwargs),
            self.time_code(self.with_cache_miss_ro, **kwargs),
            self.time_code(self.with_cache_hit, **kwargs),
        )

    def get_args(self, varying_param, value):
        args = dict(self.defaults)  # make a copy
        args[varying_param] = value
        return args

    @pytest.mark.slow
    @pytest.mark.parametrize("varying_param", ["D", "N", "M"])
    def test_cache_benchmark(self, tmpdir, varying_param, analytics, plt):
        varying = {
            "D": np.asarray(np.linspace(1, 512, 10), dtype=int),
            "N": np.asarray(np.linspace(10, 500, 8), dtype=int),
            "M": np.asarray(np.linspace(750, 2500, 8), dtype=int),
        }[varying_param]
        axis_label = self.param_to_axis_label[varying_param]

        times = [
            self.time_all(tmpdir=str(tmpdir), **self.get_args(varying_param, v))
            for v in varying
        ]

        for i, data in enumerate(zip(*times)):
            plt.plot(varying, np.median(data, axis=1), label=self.labels[i])
            plt.xlim(right=varying[-1])
            analytics.add_data(varying_param, varying, axis_label)
            analytics.add_data(self.keys[i], data)

        plt.xlabel("Number of %s" % axis_label)
        plt.ylabel("Build time (s)")
        plt.legend(loc="best")

        # TODO: add assertions

    @pytest.mark.compare
    @pytest.mark.parametrize("varying_param", ["D", "N", "M"])
    def test_compare_cache_benchmark(self, varying_param, analytics_data, plt):
        stats = pytest.importorskip("scipy.stats")

        d1, d2 = analytics_data
        assert np.all(
            d1[varying_param] == d2[varying_param]
        ), "Cannot compare different parametrizations"
        axis_label = self.param_to_axis_label[varying_param]

        logging.info("Cache, varying %s:", axis_label)
        for label, key in zip(self.labels, self.keys):
            clean_d1 = [reject_outliers(d) for d in d1[key]]
            clean_d2 = [reject_outliers(d) for d in d2[key]]
            diff = [np.median(b) - np.median(a) for a, b in zip(clean_d1, clean_d2)]

            p_values = np.array(
                [
                    2.0 * stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                    for a, b in zip(clean_d1, clean_d2)
                ]
            )
            overall_p = 1.0 - np.prod(1.0 - p_values)
            if overall_p < 0.05:
                logging.info(
                    "  %s: Significant change (p <= %.3f). See plots.",
                    label,
                    np.ceil(overall_p * 1000.0) / 1000.0,
                )
            else:
                logging.info("  %s: No significant change.", label)

            plt.plot(d1[varying_param], diff, label=label)

        plt.xlabel("Number of %s" % axis_label)
        plt.ylabel("Difference in build time (s)")
        plt.legend(loc="best")


class TestCacheShrinkBenchmark:
    n_trials = 50

    setup = dedent(
        """\
        import numpy as np
        import nengo
        import nengo.cache
        from nengo.rc import rc

        rc.set("progress", "progress_bar", "none")
        rc.set("decoder_cache", "path", {tmpdir!r})

        for i in range(10):
            model = nengo.Network(seed=i)
            with model:
                a = nengo.networks.EnsembleArray(10, 128, 1)
                b = nengo.networks.EnsembleArray(10, 128, 1)
                conn = nengo.Connection(a.output, b.input)
            with nengo.Simulator(model):
                pass

        rc.set("decoder_cache", "size", "0KB")
        cache = nengo.cache.DecoderCache()
        """
    )

    stmt = "with cache: cache.shrink()"

    @pytest.mark.slow
    def test_cache_shrink_benchmark(self, tmpdir, analytics):
        times = timeit.repeat(
            stmt=self.stmt,
            setup=self.setup.format(tmpdir=str(tmpdir)),
            number=1,
            repeat=self.n_trials,
        )
        logging.info("Shrink took a minimum of %f seconds.", np.min(times))
        logging.info("Shrink took a %f seconds on average.", np.mean(times))
        logging.info(
            "Shrink took a %f seconds on average with outliers rejected.",
            np.mean(reject_outliers(times)),
        )
        analytics.add_data("times", times)

        # TODO: add assertions

    @pytest.mark.compare
    def test_compare_cache_shrink_benchmark(self, analytics_data, plt):
        stats = pytest.importorskip("scipy.stats")

        d1, d2 = (x["times"] for x in analytics_data)
        clean_d1 = reject_outliers(d1)
        clean_d2 = reject_outliers(d2)

        diff = np.median(clean_d2) - np.median(clean_d1)

        p_value = (
            2.0 * stats.mannwhitneyu(clean_d1, clean_d2, alternative="two-sided")[1]
        )
        if p_value < 0.05:
            logging.info(
                "Significant change of %d seconds (p <= %.3f).",
                diff,
                np.ceil(p_value * 1000.0) / 1000.0,
            )
        else:
            logging.info("No significant change (%d).", diff)
        logging.info("Speed up: %s", np.median(clean_d1) / np.median(clean_d2))

        plt.scatter(np.ones_like(d1), d1, c="b")
        plt.scatter(2 * np.ones_like(d2), d2, c="g")
