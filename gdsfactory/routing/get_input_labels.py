from typing import List

from gdsfactory.add_labels import get_input_label, get_input_label_text
from gdsfactory.component import ComponentReference
from gdsfactory.port import Port
from gdsfactory.types import Label, Layer


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Layer,
    gc_port_name: str,
    get_input_label_text_function=get_input_label_text,
) -> List[Label]:
    """Returns list of labels for a list of grating coupler references.

    Args:
        io_gratings: grating coupler references
        ordered_ports: list of ordered_ports
        component_name:
        layer_label:
        gc_port_name: gc_port_name port name
        get_input_label_function:
    """
    elements = []
    for i, g in enumerate(io_gratings):
        label = get_input_label(
            port=ordered_ports[i],
            gc=g,
            gc_index=i,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
            get_input_label_text_function=get_input_label_text_function,
        )
        elements += [label]

    return elements
