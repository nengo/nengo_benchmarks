import nengo
import nengo.spa as spa
import numpy as np

use_spinnaker = False

import logging
logging.basicConfig(level=logging.DEBUG)

D = 16                  # dimensionality
mem_tau = 0.1           # memory time constant
mem_input_scale = 0.5   # input scaling on memory
test_time = 10.0         # amount of time to test memory for
test_present_time = 0.1 # amount of time for one test
answer_offset = 0.025    # reaction time

if use_spinnaker:
    answer_offset += 0.010  # spinnaker has delays due to passthroughs

model = spa.SPA()
with model:
    model.shape = spa.Buffer(D)
    model.color = spa.Buffer(D)

    model.bound = spa.Buffer(D)

    cconv = nengo.networks.CircularConvolution(n_neurons=200,
                                dimensions=D)

    nengo.Connection(model.shape.state.output, cconv.A)
    nengo.Connection(model.color.state.output, cconv.B)

    nengo.Connection(cconv.output, model.bound.state.input,
                     transform=mem_input_scale, synapse=mem_tau)

    deconv = nengo.networks.CircularConvolution(n_neurons=200,
                                dimensions=D, invert_b=True)
    deconv.label = 'deconv'

    model.query = spa.Buffer(D)

    model.result = spa.Buffer(D)

    nengo.Connection(model.bound.state.output, deconv.A)
    nengo.Connection(model.query.state.output, deconv.B)

    nengo.Connection(deconv.output, model.result.state.input,
                    transform=2)

    nengo.Connection(model.bound.state.output, model.bound.state.input,
                        synapse=mem_tau)


    vocab = model.get_output_vocab('result')
    model.cleanup = spa.AssociativeMemory([
        vocab.parse('RED').v,
        vocab.parse('BLUE').v,
        vocab.parse('CIRCLE').v,
        vocab.parse('SQUARE').v])

    model.clean_result = spa.Buffer(D)

    nengo.Connection(model.result.state.output, model.cleanup.input)
    nengo.Connection(model.cleanup.output, model.clean_result.state.input)


    stim_time = mem_tau / mem_input_scale
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
            index = int((t - stim_time) / test_present_time)
            return ['BLUE', 'RED', 'CIRCLE', 'SQUARE'][index % 4]

    model.input = spa.Input(
        shape = stim_shape,
        color = stim_color,
        query = stim_query,
        )

    probe = nengo.Probe(model.clean_result.state.output, synapse=0.02)
    probe_wm = nengo.Probe(model.bound.state.output, synapse=0.02)

if use_spinnaker:
    import nengo_spinnaker
    nengo_spinnaker.add_spinnaker_params(model.config)
    for node in model.all_nodes:
        if node.size_in == 0 and node.size_out > 0 and callable(node.output):
            model.config[node].function_of_time = True
    sim = nengo_spinnaker.Simulator(model)
else:
    sim = nengo.Simulator(model)


sim.run(stim_time * 2 + test_time)

if use_spinnaker:
    sim.close()

vocab = model.get_output_vocab('clean_result')
vals = [None] * 4
vals[0] = np.dot(sim.data[probe], vocab.parse('CIRCLE').v)
vals[1] = np.dot(sim.data[probe], vocab.parse('SQUARE').v)
vals[2] = np.dot(sim.data[probe], vocab.parse('BLUE').v)
vals[3] = np.dot(sim.data[probe], vocab.parse('RED').v)
vals = np.array(vals)

vocab_wm = model.get_output_vocab('bound')
vals_wm = [None] * 2
vals_wm[0] = np.dot(sim.data[probe_wm], vocab.parse('BLUE*CIRCLE').v)
vals_wm[1] = np.dot(sim.data[probe_wm], vocab.parse('RED*SQUARE').v)
vals_wm = np.array(vals_wm)

correct = np.zeros_like(vals)
for i, t in enumerate(sim.trange()):
    if t > stim_time * 2 + answer_offset:
        index = int((t - stim_time * 2 - answer_offset) / test_present_time)
        correct[index % 4, i] = 1.0

rmse = np.sqrt(np.mean((vals - correct).flatten()**2))
print('rmse: %g' % rmse)


import pylab
pylab.subplot(2,1,1)
#pylab.plot(sim.trange(), sim.data[probe])
pylab.plot(sim.trange(), vals.T)
pylab.plot(sim.trange(), correct.T)
pylab.subplot(2,1,2)
pylab.plot(sim.trange(), vals_wm.T)
pylab.show()



