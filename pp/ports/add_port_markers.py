""" add markers to each port

"""
import numpy as np
from phidl import device_layout as pd
from pp.layers import LAYER
import pp


def _rotate(v, m):
    return np.dot(m, v)


def add_port_marker_triangle(
    component, port, port_layer=LAYER.PORT, label_layer=LAYER.TEXT
):
    """
    # The port visualization pattern is a triangle with a right angle
    # The face opposite the right angle is the port width
    """
    p = port

    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d = p.width / 2

    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dtip = np.array([d, 0])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptip = p.position + _rotate(dtip, rot_mat)
    polygon = [p0, p1, ptip]

    component.add_label(
        text=p.name, position=p.midpoint, layer=label_layer,
    )

    component.add_polygon(polygon, layer=port_layer)


def add_port_marker_square_inside(
    component, port, port_length=0.1, port_layer=LAYER.PORT, label_layer=LAYER.TEXT
):
    """
    square inside
    """
    p = port
    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d = p.width / 2
    dx = port_length

    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([-dx, -d])
    dtopin = np.array([-dx, +d])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=port_layer)


def add_port_marker_square(
    component, port, port_length=0.1, port_layer=LAYER.PORT, label_layer=LAYER.PORT
):
    """
    half out
    """
    p = port
    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d = p.width / 2
    dx = port_length

    dbot = np.array([dx / 2, -d])
    dtop = np.array([dx / 2, d])
    dbotin = np.array([-dx / 2, -d])
    dtopin = np.array([-dx / 2, +d])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=port_layer)
    component.add_label(
        text=p.name, position=p.midpoint, layer=label_layer,
    )


def add_port_markers(
    component,
    add_port_marker_function=add_port_marker_square,
    add_device_metadata=True,
    **kwargs
):

    """ add port markers:

    - rectangle
    - triangle

    Add device recognition layer

    """
    if add_device_metadata:
        c = component
        points = [
            [c.xmin, c.ymin],
            [c.xmax, c.ymin],
            [c.xmax, c.ymax],
            [c.xmin, c.ymax],
        ]
        c.add_polygon(points, layer=LAYER.DEVREC)

    if hasattr(component, "ports") and component.ports:
        for p in component.ports.values():
            add_port_marker_function(component=component, port=p, **kwargs)


def get_optical_text(port, gc, gc_index=None, component_name=None):

    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    if component_name:
        name = component_name

    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    if isinstance(gc_index, int):
        text = "opt_{}_{}_({})_{}_{}".format(
            polarization, int(wavelength_nm), name, gc_index, port.name
        )
    else:
        text = "opt_{}_{}_({})_{}".format(
            polarization, int(wavelength_nm), name, port.name
        )

    return text


def get_input_label(
    port,
    gc,
    gc_index=None,
    gc_port_name="W0",
    layer_label=LAYER.LABEL,
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
    port, index=0, component_name=None, layer_label=LAYER.LABEL
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
    # from pp.components import mmi1x2
    # from pp.components import bend_circular
    # from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    # c = bend_circular()
    # cc = add_grating_couplers(c, layer_label=pp.LAYER.LABEL)

    # c = pp.c.waveguide()
    c = pp.c.crossing()
    # add_port_markers(c)
    pp.show(c)
