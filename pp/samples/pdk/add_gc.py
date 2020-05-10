from phidl import device_layout as pd
import pp
from pp.routing.manhattan import round_corners
from pp.rotate import rotate
from pp.add_labels import get_input_label

from import_gds import import_gds
from layers import LAYER
from waveguide import waveguide
from bend_circular import bend_circular


def gc_te1550():
    c = import_gds("ebeam_gc_te1550")
    c = rotate(c, 180)
    c.polarization = "te"
    c.wavelength = 1550
    return c


def gc_te1550_broadband():
    c = import_gds("ebeam_gc_te1550_broadband")
    return c


def gc_te1310():
    c = import_gds("ebeam_gc_te1310")
    c.polarization = "te"
    c.wavelength = 1310
    return c


def gc_tm1550():
    c = import_gds("ebeam_gc_tm1550")
    c.polarization = "tm"
    c.wavelength = 1550
    return c


def connect_strip(
    way_points=[],
    bend_factory=bend_circular,
    straight_factory=waveguide,
    bend_radius=10.0,
    wg_width=0.5,
    **kwargs,
):
    """
    Returns a deep-etched route formed by the given way_points with
    bends instead of corners and optionally tapers in straight sections.
    """
    bend90 = bend_factory(radius=bend_radius, width=wg_width)
    connector = round_corners(way_points, bend90, straight_factory)
    return connector


@pp.autoname
def taper_factory(layer=LAYER.WG, layers_cladding=[], **kwargs):
    c = pp.c.taper(layer=layer, layers_cladding=layers_cladding, **kwargs)
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
        text = f"opt_in_{polarization}_{int(wavelength_nm)}_({name})_{gc_index}_{port.name}"
    else:
        text = f"opt_in_{polarization}_{int(wavelength_nm)}_({name})_{port.name}"

    return text


gc_port_name = "W0"
layer_label = LAYER.LABEL


def get_input_labels_all(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=layer_label,
    gc_port_name=gc_port_name,
):
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


def get_input_labels(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=layer_label,
    gc_port_name=gc_port_name,
    port_index=1,
):
    if port_index == -1:
        return get_input_labels_all(
            io_gratings=io_gratings,
            ordered_ports=ordered_ports,
            component_name=component_name,
            gc_port_name=gc_port_name,
            port_index=port_index,
        )
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_optical_text(port=port, gc=gc, component_name=component_name)
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


def add_gc(
    component,
    layer_label=LAYER.LABEL,
    grating_coupler=gc_te1550,
    bend_factory=bend_circular,
    straight_factory=waveguide,
    taper_factory=taper_factory,
    route_filter=connect_strip,
    gc_port_name="W0",
    get_input_labels_function=get_input_labels,
):
    c = pp.routing.add_io_optical(
        component,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        route_filter=route_filter,
        grating_coupler=grating_coupler,
        layer_label=layer_label,
        taper_factory=taper_factory,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
    )
    c = rotate(c, -90)
    return c


if __name__ == "__main__":
    c = gc_te1550()
    # print(c.ports)
    c = add_gc(component=waveguide())
    pp.show(c)
