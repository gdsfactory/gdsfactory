"""Add labels to component ports."""

from typing import Callable, Optional, Union

import phidl.device_layout as pd

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.port import Port
from gdsfactory.types import Label, Layer


def get_input_label_text(
    port: Port,
    gc: Union[ComponentReference, Component],
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
    prefix: str = "",
    label_prefix: str = "opt",
) -> str:
    """Get text string for an optical port based on grating coupler.

    Args:
        port:
        gc: grating coupler
        gc_index: grating_coupler index, which grating_coupler we are labelling
        component_name:
        prefix: component prefix

    """
    polarization = gc.info.get("polarization") or gc.info_child.get("polarization")
    wavelength = gc.info.get("wavelength") or gc.info_child.get("wavelength")

    assert polarization in [
        "te",
        "tm",
    ], f"Not valid polarization {polarization} in [te, tm]"
    assert (
        isinstance(wavelength, (int, float)) and 0.5 < wavelength < 5.0
    ), f"{wavelength} is Not valid. Make sure it's in um"

    component_name = component_name or port.parent.info_child.name

    text = f"{label_prefix}_{polarization}_{int(wavelength*1e3)}_({prefix}{component_name})"
    if isinstance(gc_index, int):
        text += f"_{gc_index}_{port.name}"
    else:
        text = f"_{port.name}"

    return text


def get_input_label_text_loopback(prefix: str = "loopback_", **kwargs):
    return get_input_label_text(prefix=prefix, **kwargs)


def get_input_label(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    gc_port_name: str = "o1",
    layer_label: Layer = gf.LAYER.LABEL,
    component_name: Optional[str] = None,
    get_input_label_text_function=get_input_label_text,
) -> Label:
    """Returns a label with component info for a given grating coupler.
    Test equipment to extract grating coupler coordinates and match it to the component.

    Args:
        port: port to label
        gc: grating coupler reference
        gc_index: grating coupler index
        gc_port_name: name of grating coupler port
        layer_label: layer of the label
        component_name: for the label
        get_input_label_text_function: function to get input label
    """
    text = get_input_label_text_function(
        port=port, gc=gc, gc_index=gc_index, component_name=component_name
    )

    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    layer, texttype = pd._parse_layer(layer_label)
    return Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )


def get_input_label_electrical(
    port: Port,
    gc_index: int = 0,
    component_name: Optional[str] = None,
    layer_label: Layer = gf.LAYER.LABEL,
    gc: Optional[ComponentReference] = None,
) -> Label:
    """Returns a label to test component info for a given electrical port.
    This is the label used by T&M to extract grating coupler coordinates
    and match it to the component.

    Args:
        port:
        gc_index: index of the label
        component_name:
        layer_label:
        gc: ignored
    """

    if component_name:
        name = component_name
    elif isinstance(port.parent, gf.Component):
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    text = f"elec_{gc_index}_({name})_{port.name}"
    layer, texttype = pd._parse_layer(layer_label)
    label = Label(
        text=text,
        position=port.midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return label


def add_labels(
    component: Component,
    get_label_function: Callable = get_input_label_electrical,
    layer_label: Layer = gf.LAYER.LABEL,
    gc: Optional[Component] = None,
    **kwargs,
) -> Component:
    """Returns component with labels on some ports

    Args:
        component: to add labels to
        get_label_function: function to get label
        layer_label: layer_label
        gc: Optional grating coupler

    keyword Args:
        layer: port GDS layer
        prefix: with in port name
        orientation: in degrees
        width:
        layers_excluded: List of layers to exclude
        port_type: optical, electrical, ...
        clockwise: if True, sort ports clockwise, False: counter-clockwise

    Returns:
        original component with labels

    """
    ports = component.get_ports_list(**kwargs)

    for i, port in enumerate(ports):
        label = get_label_function(
            port=port,
            gc=gc,
            gc_index=i,
            component_name=component.name,
            layer_label=layer_label,
        )
        component.add(label)

    return component


if __name__ == "__main__":
    c = gf.c.mzi_phase_shifter()
    # add_labels_ports(c, c.get_ports_list(port_type="electrical"), prefix="pad_")
    # from gdsfactory.tests.test_labels import test_add_labels_electrical
    # c = test_add_labels_optical()
    # c = test_add_labels_electrical()
    c = gf.routing.add_fiber_single(c)
    c.show()
