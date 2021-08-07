"""Add grating_couplers to a component."""
from typing import Callable, List, Tuple

from phidl.device_layout import Label

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler.elliptical_trenches import (
    grating_coupler_te,
    grating_coupler_tm,
)
from gdsfactory.routing.get_input_labels import get_input_labels
from gdsfactory.types import ComponentFactory


@cell
def add_grating_couplers(
    component: Component,
    grating_coupler: ComponentFactory = grating_coupler_te,
    layer_label: Tuple[int, int] = gf.LAYER.LABEL,
    gc_port_name: str = "W0",
    get_input_labels_function: Callable[..., List[Label]] = get_input_labels,
) -> Component:
    """Returns new component with grating couplers and labels.

    Args:
        Component: to add grating_couplers
        grating_couplers: grating_coupler function
        layer_label: for label
        gc_port_name: where to add label
        get_input_labels_function: function to get label

    """

    cnew = Component()
    cnew.add_ref(component)
    grating_coupler = gf.call_if_func(grating_coupler)

    io_gratings = []
    optical_ports = component.get_ports_list(port_type="optical")
    for port in optical_ports:
        gc_ref = grating_coupler.ref()
        gc_port = gc_ref.ports[gc_port_name]
        gc_ref.connect(gc_port, port)
        io_gratings.append(gc_ref)
        cnew.add(gc_ref)

    labels = get_input_labels_function(
        io_gratings,
        list(component.ports.values()),
        component_name=component.name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    cnew.add(labels)
    return cnew


def add_te(*args, **kwargs):
    return add_grating_couplers(*args, **kwargs)


def add_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_grating_couplers(*args, grating_coupler=grating_coupler, **kwargs)


if __name__ == "__main__":
    # from gdsfactory.add_labels import get_optical_text
    # c = gf.components.grating_coupler_elliptical_te()
    # print(c.wavelength)

    # print(c.get_property('wavelength'))

    c = gf.components.straight(width=2)
    # cc = add_grating_couplers(c)
    cc = add_tm(c)
    print(cc)
    cc.show()
