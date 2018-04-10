****************
nengo_benchmarks
****************

Benchmark models for nengo

These benchmarks use pytry, available via ``pip install pytry``
or from pytry_

There are three ways you can run these benchmarks.  First, from
the command line:

.. code-block:: bash

   pytry nengo_benchmarks/comm_channel.py

There are command-line options to adjust parameters of the benchmark.
To list these command-line options, do ``--help``

You can also run these benchmarks in the nengo_gui by doing:

.. code-block:: bash

   pytry nengo_benchmarks/comm_channel.py --gui

Finally, you can run the benchmarks programmatically via:

.. code-block:: python

   import nengo_benchmarks
   result = nengo_benchmarks.CommunicationChannel().run()

When running benchmarks this way, you can specify parameters
as arguments to the ``run()`` function

See pytry_ for more details

.. _pytry: https://github.com/tcstewar/pytry
