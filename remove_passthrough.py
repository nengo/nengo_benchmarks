import nengo

import nengo.utils.builder

def remove_passthrough_nodes(network, keep_probes=True):
    m = nengo.Network()

    conns = list(network.all_connections)
    inputs, outputs = nengo.utils.builder.find_all_io(conns)

    keep_nodes = []
    if keep_probes:
        for probe in network.all_probes:
            if isinstance(probe.target, nengo.Node):
                if probe.target.output is None:
                    keep_nodes.append(probe.target)

    with m:
        for ens in network.all_ensembles:
            m.add(ens)
        for node in network.all_nodes:
            if node.output is None and node not in keep_nodes:
                conns_in = inputs[node]
                conns_out = outputs[node]

                success = True

                new_conns = []
                for c_in in conns_in:
                    if not success:
                        break
                    for c_out in conns_out:
                        try:
                            c = nengo.utils.builder._create_replacement_connection(c_in, c_out)
                            if c is not None:
                                new_conns.append(c)
                        except NotImplementedError:
                            print 'could not remove', node, c_in, c_out
                            success = False
                            break
                if success:
                    for c in new_conns:
                        conns.append(c)
                        outputs[c.pre_obj].append(c)
                        inputs[c.post_obj].append(c)
                    for c in conns_in:
                        conns.remove(c)
                        outputs[c.pre_obj].remove(c)
                    for c in conns_out:
                        conns.remove(c)
                        inputs[c.post_obj].remove(c)
            else:
                m.add(node)
        for conn in conns:
            m.add(conn)
        if keep_probes:
            for probe in network.all_probes:
                m.add(probe)


    return m
