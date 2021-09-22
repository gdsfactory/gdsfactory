"""Add labels to component ports for lab measurements
"""

from typing import Callable, Optional, Union

import phidl.device_layout as pd
from phidl.device_layout import Label

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.port import Port
from gdsfactory.types import Layer


def get_input_label_text(
    port: Port,
    gc: Union[ComponentReference, Component],
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
    prefix: str = "",
) -> str:
    """Get text string for an optical port."""
    polarization = gc.get_property("polarization")
    wavelength = gc.get_property("wavelength")
    prefix = prefix or ""

    assert polarization in [
        "te",
        "tm",
    ], f"Not valid polarization {polarization} in [te, tm]"
    assert (
        isinstance(wavelength, (int, float)) and 0.5 < wavelength < 5.0
    ), f"{wavelength} is Not valid. Make sure it's in um"

    component_name = component_name or port.parent.get_property("name")

    text = f"opt_{polarization}_{int(wavelength*1e3)}_({prefix}{component_name})"
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
    """
    text = get_input_label_text_function(
        port=port, gc=gc, gc_index=gc_index, component_name=component_name
    )

    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    layer, texttype = pd._parse_layer(layer_label)
    return pd.Label(
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
    label = pd.Label(
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
    """Add labels a particular type of ports

    Args:
        component: to add labels to
        get_label_function: function to get label
        layer_label: layer_label
        **port_settings to select

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


def test_optical_labels() -> Component:
    c = gf.components.straight()
    gc = gf.components.grating_coupler_elliptical_te()
    label1 = get_input_label(
        port=c.ports["o1"], gc=gc, gc_index=0, layer_label=gf.LAYER.LABEL
    )
    label2 = get_input_label(
        port=c.ports["o2"], gc=gc, gc_index=1, layer_label=gf.LAYER.LABEL
    )
    add_labels(c, get_label_function=get_input_label, gc=gc)
    labels_text = [c.labels[0].text, c.labels[1].text]
    # print(label1)
    # print(label2)

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


def test_electrical_labels() -> Component:
    c = gf.components.wire_straight()
    label1 = get_input_label_electrical(
        port=c.ports["e1"], layer_label=gf.LAYER.LABEL, gc_index=0
    )
    label2 = get_input_label_electrical(
        port=c.ports["e2"], layer_label=gf.LAYER.LABEL, gc_index=1
    )
    add_labels(component=c, get_label_function=get_input_label_electrical)
    labels_text = [c.labels[0].text, c.labels[1].text]

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


if __name__ == "__main__":
    # c = test_optical_labels()
    c = test_electrical_labels()
    c.show()
