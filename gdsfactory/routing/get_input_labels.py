from __future__ import annotations

from functools import partial

from gdsfactory.add_labels import (
    get_input_label,
    get_input_label_text,
    get_input_label_text_dash,
)
from gdsfactory.component import ComponentReference
from gdsfactory.port import Port
from gdsfactory.typings import Label, LayerSpec


def get_input_labels(
    io_gratings: list[ComponentReference],
    ordered_ports: list[Port],
    component_name: str,
    layer_label: LayerSpec,
    gc_port_name: str,
    get_input_label_text_function=get_input_label_text,
) -> list[Label]:
    """Returns list of labels for a list of grating coupler references.

    Args:
        io_gratings: grating coupler references.
        ordered_ports: list of ordered_ports.
        component_name: component name.
        layer_label: layer spec for the label.
        gc_port_name: gc_port_name port name.
        get_input_label_function: function to get input label.

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


get_input_labels_dash = partial(
    get_input_labels,
    get_input_label_text_function=get_input_label_text_dash,
)
