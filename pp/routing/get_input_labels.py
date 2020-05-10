from pp.add_labels import get_input_label


def get_input_labels(
    io_gratings, ordered_ports, component_name, layer_label, gc_port_name
):
    elements = []
    for i, g in enumerate(io_gratings):
        label = get_input_label(
            port=ordered_ports[i],
            gc=g,
            gc_index=i,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
        elements += [label]

    return elements
