"""SiEPIC labels one grating coupler from the fiber array using a GDS label."""

from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_layout import Label
from gdsfactory.components import grating_coupler_te
from gdsfactory.components.straight import straight
from gdsfactory.port import Port
from gdsfactory.typings import (
    ComponentReference,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
)


def get_input_label_text(
    port: Port,
    gc: ComponentReference,
    gc_index: int | None = None,
    component_name: str | None = None,
    username: str = "YourUserName",
) -> str:
    """Return label for port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        gc_index: grating coupler index.
        component_name: optional component name.
        username: for the label.
    """
    polarization = gc.info.get("polarization")
    wavelength = gc.info.get("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()!r} in [TE, TM]"
    assert (
        isinstance(wavelength, int | float) and 1.0 < wavelength < 2.0
    ), f"{wavelength} is Not valid 1000 < wavelength < 2000"

    name = component_name or port.parent.metadata_child.get("name")
    return f"opt_in_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}_({name})-{gc_index}-{port.name}"


def get_input_labels(
    io_gratings: list[ComponentReference],
    ordered_ports: list[Port],
    component_name: str,
    layer_label: tuple[int, int] = (10, 0),
    gc_port_name: str = "o1",
    port_index: int = 1,
    get_input_label_text_function: Callable = get_input_label_text,
) -> list[Label]:
    """Return list of labels for all component ports.

    Args:
        io_gratings: list of grating_coupler references.
        ordered_ports: list of ports.
        component_name: name.
        layer_label: for the label.
        gc_port_name: grating_coupler port.
        port_index: index of the port.
        get_input_label_text_function: function to get input label.

    """
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_input_label_text_function(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = gf.get_layer(layer_label)
    label = Label(
        text=text,
        origin=gc.ports[gc_port_name].center,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@cell
def add_fiber_array_siepic(
    component: ComponentSpec = straight,
    component_name: str | None = None,
    gc_port_name: str = "o1",
    get_input_labels_function: Callable = get_input_labels,
    with_loopback: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    cross_section: CrossSectionSpec = "strip",
    layer_label: LayerSpec = (10, 0),
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        component_name: for the label.
        gc_port_name: grating coupler input port name 'o1'.
        get_input_labels_function: function to get input labels for grating couplers.
        with_loopback: True, adds loopback structures.
        optical_routing_type: None: autoselection, 0: no extension.
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        layer_label: for label.

    """
    c = gf.Component()

    component = gf.routing.add_fiber_array(
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        get_input_label_text_function=get_input_label_text,
        with_loopback=with_loopback,
        optical_routing_type=optical_routing_type,
        layer_label=layer_label,
        fanout_length=fanout_length,
        cross_section=cross_section,
        **kwargs,
    )
    ref = c << component
    ref.rotate(-90)
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = add_fiber_array_siepic()
    c.show(show_ports=True)
