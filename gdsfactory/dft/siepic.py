"""SiEPIC labels one grating coupler from the fiber array using a GDS label
(not fabricated) """

from typing import Callable, List, Optional, Tuple

from phidl import device_layout as pd
from phidl.device_layout import Label

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components import grating_coupler_te
from gdsfactory.components.straight import straight
from gdsfactory.port import Port
from gdsfactory.types import ComponentReference, ComponentSpec, CrossSectionSpec, Layer


def get_input_label_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
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
    ], f"Not valid polarization {polarization.upper()} in [TE, TM]"
    assert (
        isinstance(wavelength, (int, float)) and 1.0 < wavelength < 2.0
    ), f"{wavelength} is Not valid 1000 < wavelength < 2000"

    name = component_name or port.parent.metadata_child.get("name")
    # name = component_name
    # elif type(port.parent) == Component:
    # name = port.parent.name
    # else:
    # name = port.parent.ref_cell.name
    # name = name.replace("_", "-")

    label = (
        f"opt_in_{polarization.upper()}_{int(wavelength*1e3)}_device_"
        f"{username}_({name})-{gc_index}-{port.name}"
    )
    return label


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = (10, 0),
    gc_port_name: str = "o1",
    port_index: int = 1,
    get_input_label_text_function: Callable = get_input_label_text,
) -> List[Label]:
    """Return list of labels for all component ports.

    Args:
        io_gratings:
        ordered_ports:
        component_name:
        layer_label:
        gc_port_name:
        port_index:
        get_input_label_text_function:
    """
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_input_label_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@cell
def add_fiber_array_siepic(
    component: ComponentSpec = straight,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    get_input_labels_function: Callable = get_input_labels,
    with_loopback: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    cross_section: CrossSectionSpec = "strip",
    layer_label: Layer = (10, 0),
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
    c.show()
