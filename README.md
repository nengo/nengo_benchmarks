# nengo_benchmarks
Benchmark models for nengo

These benchmarks use pytry, available via `pip install pytry` or
from https://github.com/tcstewar/pytry

There are three ways you can run these benchmarks.  First, from
the command line:

    pytry nengo_benchmarks/comm_channel.py

There are command-line options to adjust parameters of the benchmark.
To list these command-line options, do `--help`

You can also run these benchmarks in the nengo_gui by doing

    pytry nengo_benchmarks/comm_channel.py --gui

Finally, you can run the benchmarks programmatically via:

    import nengo_benchmarks
    result = nengo_benchmarks.CommunicationChannel().run()

When running benchmarks this way, you can specify parameters
as arguments to the `run()` function

See https://github.com/tcstewar/pytry for more details
