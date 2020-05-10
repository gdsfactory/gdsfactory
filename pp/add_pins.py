""" add markers to each port

"""
import numpy as np
from pp.layers import LAYER
import pp


def _rotate(v, m):
    return np.dot(m, v)


def add_pin_triangle(component, port, port_layer=LAYER.PORT, label_layer=LAYER.TEXT):
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


def add_pin_square_inside(
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


def add_pin_square(
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
        text=str(p.name), position=p.midpoint, layer=label_layer,
    )


def add_pins(
    component, add_port_marker_function=add_pin_square, add_outline=True, **kwargs,
):

    """ add port markers:

    - rectangle
    - triangle

    Add device recognition layer

    """
    if add_outline:
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


if __name__ == "__main__":
    # from pp.components import mmi1x2
    # from pp.components import bend_circular
    # from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    # c = bend_circular()
    # cc = add_grating_couplers(c, layer_label=pp.LAYER.LABEL)

    # c = pp.c.waveguide()
    c = pp.c.crossing()
    # add_pins(c)
    pp.show(c)
