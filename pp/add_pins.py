"""add_pin adss a Pin to a port, add_pins adds Pins to all ports:

- pins
- outline

"""
from typing import Optional, Tuple, Callable, List
import json
import numpy as np
from pp.layers import LAYER, port_type2layer
from pp.port import read_port_markers
import pp
from pp.container import container
from pp.component import Component, ComponentReference


def _rotate(v, m):
    return np.dot(m, v)


def add_pin_triangle(component, port, layer=LAYER.PORT, label_layer=LAYER.TEXT):
    """add triangle pin with a right angle, pointing out of the port"""
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
    """add square pin towards the inside of the port

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
    component,
    port,
    port_length=0.1,
    layer=LAYER.PORT,
    label_layer=LAYER.PORT,
    port_margin=0,
):
    """add half out pin to a component

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

    d = p.width / 2 + port_margin
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


def add_outline(component, reference: ComponentReference, layer=LAYER.DEVREC):
    """Adds devices outline bounding box in layer."""
    c = reference
    points = [
        [c.xmin, c.ymin],
        [c.xmax, c.ymin],
        [c.xmax, c.ymax],
        [c.xmin, c.ymax],
    ]
    component.add_polygon(points, layer=layer)


def add_pins(
    component,
    reference: ComponentReference,
    add_port_marker_function=add_pin_square,
    port_type2layer=port_type2layer,
    **kwargs,
):
    """add port markers:

    Args:
        component: to add ports
        reference: take ports from a reference
        add_port_marker_function:
        port_type2layer: dict mapping port types to marker layers for ports

    """

    if hasattr(reference, "ports") and reference.ports:
        for p in reference.ports.values():
            layer = port_type2layer[p.port_type]
            add_port_marker_function(
                component=component, port=p, layer=layer, label_layer=layer, **kwargs
            )


def add_settings_label(
    component,
    reference: Optional[ComponentReference] = None,
    label_layer=LAYER.LABEL_SETTINGS,
):
    """Add settings in label, ignores component.ignore keys."""
    reference = reference or component
    settings = reference.get_settings()
    settings_string = f"settings={json.dumps(settings, indent=2)}"
    component.add_label(
        position=reference.center, text=settings_string, layer=label_layer
    )


def add_instance_label(
    component: Component,
    reference: ComponentReference,
    instance_name: Optional[str] = None,
    layer: Tuple[int, int] = LAYER.LABEL_INSTANCE,
):
    """Adds label."""
    instance_name = (
        instance_name
        or f"{reference.parent.name},{int(reference.x)},{int(reference.y)}"
    )
    component.add_label(
        text=instance_name, position=(reference.x, reference.y), layer=layer
    )


def add_markers(
    component: Component,
    reference: ComponentReference,
    add_outline_function: Optional[Callable] = add_outline,
    add_pins_function: Optional[Callable] = add_pins,
    add_settings_function: Optional[Callable] = add_settings_label,
    add_instance_label_function: Optional[Callable] = add_instance_label,
):
    """Add markers:
    - outline
    - pins for the ports
    - label for the name
    - label for the settings

    Args:
        component: where to add the markers
        reference:
        pins_function: function to add pins to ports
        add_outline_function: function to add outline around the device

    """
    from pp.component import Component, ComponentReference

    assert isinstance(component, Component), f"{component} needs to be a Component"
    assert isinstance(
        reference, ComponentReference
    ), f"{reference} needs to be a ComponentReference"
    if add_outline_function:
        add_outline_function(component=component, reference=reference)
    if add_pins_function:
        add_pins_function(component=component, reference=reference)
    if add_settings_function:
        add_settings_function(component=component, reference=reference)
    if add_instance_label_function:
        add_instance_label_function(component=component, reference=reference)


@container
def add_pins_container(
    component: Component, function: Callable = add_markers, recursive: bool = False
):
    """Add pins to a Component and returns a container

    Args:
        component:
        function: function to add pins
        recursive: goes down the hierarchy
    """

    c = pp.Component(f"{component.name}_pins")
    reference = c << component
    function(component=c, reference=reference)

    if recursive:
        for reference in component.references:
            function(component=c, reference=reference)

    return c


def add_pins_to_references(
    component: Component,
    references: Optional[List[ComponentReference]] = None,
    function: Callable = add_markers,
):
    """Add pins to a Component.

    Args:
        component: t
        references: list of references, taken from component by default
        function: function to add pins
    """
    references = references or component.references
    for r in references:
        function(component=component, reference=r)


def add_pins_triangle(component, add_port_marker_function=add_pin_triangle, **kwargs):
    return add_pins(
        component=component, add_port_marker_function=add_port_marker_function, **kwargs
    )


def test_add_pins():
    c1 = pp.c.mzi2x2(with_elec_connections=True)
    c2 = add_pins_container(c1)

    n_optical_expected = 4
    n_dc_expected = 3
    # polygons = 194

    port_layer_optical = port_type2layer["optical"]
    port_markers_optical = read_port_markers(c2, [port_layer_optical])
    n_optical = len(port_markers_optical.polygons)

    port_layer_dc = port_type2layer["dc"]
    port_markers_dc = read_port_markers(c2, [port_layer_dc])
    n_dc = len(port_markers_dc.polygons)

    print(len(c1.get_polygons()))
    print(len(c2.get_polygons()))
    print(n_optical)
    print(n_dc)

    # assert len(c1.get_polygons()) == polygons
    # assert len(c2.get_polygons()) == polygons + 41
    assert n_optical == n_optical_expected
    assert n_dc_expected == n_dc_expected


if __name__ == "__main__":
    # c = pp.c.waveguide()
    # cc = add_pins_container(component=c)
    # pp.show(cc)

    test_add_pins()
    # from pp.components import mmi1x2
    # from pp.components import bend_circular
    # from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    # cc = add_grating_couplers(c, layer_label=pp.LAYER.LABEL)

    # c = pp.c.waveguide()
    # c = pp.c.crossing()
    # c = pp.c.bend_circular()
    # add_pins(c)
    # pp.show(c)
