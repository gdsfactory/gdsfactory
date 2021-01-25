"""Add labels to component ports for lab measurements
"""

from typing import Callable, Optional, Union

import phidl.device_layout as pd
from phidl.device_layout import Label

import pp
from pp.component import Component, ComponentReference
from pp.port import Port
from pp.types import Layer


def get_optical_text(
    port: Port,
    gc: Union[ComponentReference, Component],
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
) -> str:
    """Get test and measurement label for an optical port"""
    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    assert polarization in [
        "te",
        "tm",
    ], f"Not valid polarization {polarization} in [te, tm]"
    assert (
        isinstance(wavelength_nm, (int, float)) and 1000 < wavelength_nm < 2000
    ), f"{wavelength_nm} is Not valid 1000 < wavelength < 2000"

    if component_name:
        name = component_name

    elif isinstance(port.parent, pp.Component):
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    if isinstance(gc_index, int):
        text = (
            f"opt_{polarization}_{int(wavelength_nm)}_({name})_{gc_index}_{port.name}"
        )
    else:
        text = f"opt_{polarization}_{int(wavelength_nm)}_({name})_{port.name}"

    return text


def get_input_label(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    gc_port_name: str = "W0",
    layer_label: Layer = pp.LAYER.LABEL,
    component_name: Optional[str] = None,
) -> Label:
    """
    Generate a label with component info for a given grating coupler.
    This is the label used by T&M to extract grating coupler coordinates
    and match it to the component.
    """
    text = get_optical_text(
        port=port, gc=gc, gc_index=gc_index, component_name=component_name
    )

    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return label


def get_input_label_electrical(
    port: Port,
    gc_index: int = 0,
    component_name: Optional[str] = None,
    layer_label: Layer = pp.LAYER.LABEL,
    gc: Optional[ComponentReference] = None,
) -> Label:
    """
    Generate a label to test component info for a given electrical port.
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

    elif isinstance(port.parent, pp.Component):
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    text = f"elec_{gc_index}_({name})_{port.name}"
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text, position=port.midpoint, anchor="o", layer=layer, texttype=texttype,
    )
    return label


def add_labels(
    component: Component,
    port_type: str = "dc",
    get_label_function: Callable = get_input_label_electrical,
    layer_label: Layer = pp.LAYER.LABEL,
    gc: Optional[Component] = None,
) -> Component:
    """Add labels a particular type of ports

    Args:
        component: to add labels to
        port_type: type of port ('dc', 'optical', 'electrical')
        get_label_function: function to get label
        layer_label: layer_label

    Returns:
        original component with labels

    """
    ports = component.get_ports_list(port_type=port_type)

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


def test_optical_labels() -> Component:
    c = pp.c.waveguide()
    gc = pp.c.grating_coupler_elliptical_te()
    label1 = get_input_label(
        port=c.ports["W0"], gc=gc, gc_index=0, layer_label=pp.LAYER.LABEL
    )
    label2 = get_input_label(
        port=c.ports["E0"], gc=gc, gc_index=1, layer_label=pp.LAYER.LABEL
    )
    add_labels(c, port_type="optical", get_label_function=get_input_label, gc=gc)
    labels_text = [c.labels[0].text, c.labels[1].text]
    # print(label1)
    # print(label2)

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


def test_electrical_labels() -> Component:
    c = pp.c.wire()
    label1 = get_input_label_electrical(
        port=c.ports["E_1"], layer_label=pp.LAYER.LABEL, gc_index=0
    )
    label2 = get_input_label_electrical(
        port=c.ports["E_0"], layer_label=pp.LAYER.LABEL, gc_index=1
    )
    add_labels(c, port_type="dc", get_label_function=get_input_label_electrical)
    labels_text = [c.labels[0].text, c.labels[1].text]

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


if __name__ == "__main__":
    # c = test_optical_labels()
    c = test_electrical_labels()
    pp.show(c)
