import nengo

import numpy as np

import remove_passthrough

class NengoInfo(object):
    def __init__(self, network):
        self.dimensions = []
        self.n_neurons = []
        self.n_neurons_times_dimensions = []

        network = remove_passthrough.remove_passthrough_nodes(network,
                    keep_probes=False)

        inputs, outputs = nengo.utils.builder.find_all_io(
                                network.all_connections)

        self.decoder_sizes = {}

        for ens in network.all_ensembles:
            if isinstance(ens.neuron_type, nengo.Direct):
                print 'ignoring', ens
            else:
                self.dimensions.append(ens.dimensions)
                self.n_neurons.append(ens.n_neurons)
                self.n_neurons_times_dimensions.append(ens.dimensions * ens.n_neurons)
            if ens in outputs:
                functions = {}
                for conn in outputs[ens]:
                    key = (str(conn.pre_slice), str(conn.function))
                    if key not in functions:
                        functions[key] = [conn]
                    else:
                        functions[key].append(conn)
                n_decoders = 0
                for key, conns in functions.items():
                    n_decoders += conns[0].size_mid
                self.decoder_sizes[ens] = n_decoders * ens.n_neurons




        self.transform_sizes = []
        for conn in network.all_connections:
            t = conn.transform
            if len(t.shape) == 0:
                pass
            elif len(t.shape) == 1:
                pass
            elif isinstance(conn.post_obj, nengo.ensemble.Neurons):
                pass
            else:
                self.transform_sizes.append(t.shape[0] * t.shape[1])
                if self.transform_sizes[-1] > 10000:
                    print 'large transform', conn, conn.transform.shape


    def print_info(self):
        print 'Ensembles:'
        print '  mean dimensionality per neuron', float(np.sum(self.n_neurons_times_dimensions)) / np.sum(self.n_neurons)
        print '  total number of neurons', np.sum(self.n_neurons)
        print '  total decoder size', np.sum(self.decoder_sizes.values())

        print 'Connections:'
        print '  total transform size', np.sum(self.transform_sizes)






