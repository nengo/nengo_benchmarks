import nengo
import nengo.spa as spa
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

D = 16

model = spa.SPA()
with model:
    model.shape = spa.Buffer(D)
    model.color = spa.Buffer(D)

    model.bound = spa.Buffer(D)

    cconv = nengo.networks.CircularConvolution(n_neurons=200,
                                dimensions=16)

    nengo.Connection(model.shape.state.output, cconv.A)
    nengo.Connection(model.color.state.output, cconv.B)

    nengo.Connection(cconv.output, model.bound.state.input)

    deconv = nengo.networks.CircularConvolution(n_neurons=200,
                                dimensions=16, invert_b=True)
    deconv.label = 'deconv'

    model.query = spa.Buffer(D)

    model.result = spa.Buffer(D)

    nengo.Connection(model.bound.state.output, deconv.A)
    nengo.Connection(model.query.state.output, deconv.B)

    nengo.Connection(deconv.output, model.result.state.input,
                    transform=2)

    nengo.Connection(model.bound.state.output, model.bound.state.input,
                        synapse=0.1)


    vocab = model.get_output_vocab('result')
    model.cleanup = spa.AssociativeMemory([
        vocab.parse('RED').v,
        vocab.parse('BLUE').v,
        vocab.parse('CIRCLE').v,
        vocab.parse('SQUARE').v])

    model.clean_result = spa.Buffer(D)

    nengo.Connection(model.result.state.output, model.cleanup.input)
    nengo.Connection(model.cleanup.output, model.clean_result.state.input)


    stim_time = 0.05
    def stim_color(t):
        if 0 < t < stim_time:
            return 'BLUE'
        elif stim_time < t < stim_time*2:
            return 'RED'
        else:
            return '0'

    def stim_shape(t):
        if 0 < t < stim_time:
            return 'CIRCLE'
        elif stim_time < t < stim_time*2:
            return 'SQUARE'
        else:
            return '0'

    def stim_query(t):
        if t < stim_time*2:
            return '0'
        else:
            index = int((t - stim_time) / 0.1)
            return ['BLUE', 'RED', 'CIRCLE', 'SQUARE'][index % 4]

    model.input = spa.Input(
        shape = stim_shape,
        color = stim_color,
        query = stim_query,
        )

    probe = nengo.Probe(model.clean_result.state.output, synapse=0.02)

import nengo_spinnaker
nengo_spinnaker.add_spinnaker_params(model.config)
for node in model.all_nodes:
    if node.size_in == 0 and node.size_out > 0 and callable(node.output):
        model.config[node].function_of_time = True
sim = nengo_spinnaker.Simulator(model)


sim.run(1)

vocab = model.get_output_vocab('clean_result')
vals = [None] * 4
vals[0] = np.dot(sim.data[probe], vocab.parse('CIRCLE').v)
vals[1] = np.dot(sim.data[probe], vocab.parse('SQUARE').v)
vals[2] = np.dot(sim.data[probe], vocab.parse('BLUE').v)
vals[3] = np.dot(sim.data[probe], vocab.parse('RED').v)
vals = np.array(vals)

import pylab
#pylab.plot(sim.trange(), sim.data[probe])
pylab.plot(sim.trange(), vals.T)
pylab.show()



