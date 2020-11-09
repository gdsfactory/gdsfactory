""" add markers to Devices:

- pins
- outline

"""
import numpy as np
from pp.layers import LAYER, port_type2layer
from pp.port import read_port_markers
import pp


def _rotate(v, m):
    return np.dot(m, v)


def add_pin_triangle(component, port, layer=LAYER.PORT, label_layer=LAYER.TEXT):
    """ add triangle pin with a right angle, pointing out of the port
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

    component.add_polygon(polygon, layer=layer)


def add_pin_square_inside(
    component, port, port_length=0.1, layer=LAYER.PORT, label_layer=LAYER.TEXT
):
    """ add square pin towards the inside of the port

    .. code::
           _______________
          |               |
          |               |
          |               |
          ||              |
          ||              |
          |               |
          |      __       |
          |_______________|


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
    component.add_polygon(polygon, layer=layer)


def add_pin_square(
    component, port, port_length=0.1, layer=LAYER.PORT, label_layer=LAYER.PORT
):
    """ add half out pin to a component

    .. code::
           _______________
          |               |
          |               |
          |               |
         |||              |
         |||              |
          |               |
          |      __       |
          |_______________|
                 __

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
    component.add_polygon(polygon, layer=layer)

    component.add_label(
        text=str(p.name), position=p.midpoint, layer=label_layer,
    )


def add_outline(component, layer=LAYER.DEVREC):
    """ adds devices outline in layer """
    c = component
    points = [
        [c.xmin, c.ymin],
        [c.xmax, c.ymin],
        [c.xmax, c.ymax],
        [c.xmin, c.ymax],
    ]
    c.add_polygon(points, layer=layer)


def add_pins(
    component,
    add_port_marker_function=add_pin_square,
    port_type2layer=port_type2layer,
    **kwargs,
):
    """ add port markers:

    Args:
        component: to add ports
        add_port_marker_function:
        port_type2layer: dict mapping port types to marker layers for ports

    """

    if hasattr(component, "ports") and component.ports:
        for p in component.ports.values():
            layer = port_type2layer[p.port_type]
            add_port_marker_function(
                component=component, port=p, layer=layer, label_layer=layer, **kwargs
            )


def add_pins_and_outline(
    component, pins_function=add_pins, add_outline_function=add_outline
):
    add_outline_function(component)
    pins_function(component)


def add_pins_triangle(component, add_port_marker_function=add_pin_triangle, **kwargs):
    return add_pins(
        component=component, add_port_marker_function=add_port_marker_function, **kwargs
    )


def test_add_pins():
    component = pp.c.mzi2x2(with_elec_connections=True)
    # print(len(component.get_polygons()))
    assert len(component.get_polygons()) == 194

    add_pins(component)
    assert len(component.get_polygons()) == 194 + 7

    # print(len(component.get_polygons()))

    port_layer = port_type2layer["optical"]
    port_markers = read_port_markers(component, [port_layer])
    assert len(port_markers.polygons) == 4

    port_layer = port_type2layer["dc"]
    port_markers = read_port_markers(component, [port_layer])
    assert len(port_markers.polygons) == 3

    # for port_layer, port_type in port_layer2type.items():
    #     port_markers = read_port_markers(component, [port_layer])
    #     print(len(port_markers.polygons))


if __name__ == "__main__":
    test_add_pins()
    # from pp.components import mmi1x2
    # from pp.components import bend_circular
    # from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    # c = bend_circular()
    # cc = add_grating_couplers(c, layer_label=pp.LAYER.LABEL)

    # c = pp.c.waveguide()
    # c = pp.c.crossing(pins=True)
    # add_pins(c)
    # pp.show(c)
