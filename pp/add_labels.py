"""" add electrical labels to each device port

"""

import phidl.device_layout as pd
import pp


def add_label(component, text, position=(0, 0), layer=pp.LAYER.LABEL):
    gds_layer_label, gds_datatype_label = pd._parse_layer(layer)
    label = pd.Label(
        text=text,
        position=position,
        anchor="o",
        layer=gds_layer_label,
        texttype=gds_datatype_label,
    )
    component.add(label)
    return component


def add_labels(component):
    c = pp.Component()
    electrical_ports = component.get_electrical_ports()
    c.add(component.ref())

    for i, port in enumerate(electrical_ports):
        label = get_input_label_electrical(port, i, component_name=component.name)
        c.add(label)

    return c


def get_optical_text(port, gc, gc_index=None, component_name=None):
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

    elif type(port.parent) == pp.Component:
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
    port,
    gc,
    gc_index=None,
    gc_port_name="W0",
    layer_label=pp.LAYER.LABEL,
    component_name=None,
):
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
    port, index=0, component_name=None, layer_label=pp.LAYER.LABEL
):
    """
    Generate a label to test component info for a given grating coupler.
    This is the label used by T&M to extract grating coupler coordinates
    and match it to the component.
    """

    if component_name:
        name = component_name

    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    text = "elec_{}_({})_{}".format(index, name, port.name)

    layer, texttype = pd._parse_layer(layer_label)

    label = pd.Label(
        text=text, position=port.midpoint, anchor="o", layer=layer, texttype=texttype,
    )
    return label


def _demo_input_label():
    c = pp.c.bend_circular()
    gc = pp.c.grating_coupler_elliptical_te()
    label = get_input_label(port=c.ports["W0"], gc=gc, layer_label=pp.LAYER.LABEL)
    print(label)


if __name__ == "__main__":
    from pp.components.electrical.pad import pad

    c = pad(width=10, height=10)
    print(c.ports)
    c2 = add_labels(c)
    pp.show(c2)
