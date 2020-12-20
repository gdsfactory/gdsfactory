from typing import List

from omegaconf.listconfig import ListConfig
from phidl.device_layout import Label

from pp.add_labels import get_input_label
from pp.component import ComponentReference
from pp.port import Port


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: ListConfig,
    gc_port_name: str,
) -> List[Label]:
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
