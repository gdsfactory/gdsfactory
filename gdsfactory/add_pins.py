"""add_pin adss a Pin to a port, add_pins adds Pins to all ports:

- pins
- outline

Some functions modify a component without changing its name.
Make sure these functions are inside a new Component.
They modify the geometry of a component (add pins, labels, grating couplers ...) without modifying the cell name

"""
import json
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component, ComponentReference
from gdsfactory.port import Port, read_port_markers
from gdsfactory.tech import LAYER, PORT_TYPE_TO_LAYER
from gdsfactory.types import Layer


def _rotate(v: ndarray, m: ndarray) -> ndarray:
    return np.dot(m, v)


def add_pin_triangle(
    component: Component,
    port: Port,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add triangle pin with a right angle, pointing out of the port

    Args:
        component:
        port: Port
        layer: for the pin marker
        label_layer: for the label
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
    component.add_polygon(polygon, layer=layer)

    if label_layer:
        component.add_label(
            text=p.name,
            position=p.midpoint,
            layer=label_layer,
        )


def add_pin_square_inside(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add square pin towards the inside of the port

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        label_layer: for the label

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

    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([-pin_length, -d])
    dtopin = np.array([-pin_length, +d])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)
    if label_layer:
        component.add_label(
            text=str(p.name),
            position=p.midpoint,
            layer=label_layer,
        )


def add_pin_square_double(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add two square pins: one inside with label, one outside

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        label_layer: for the label

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

    # outer square
    d = p.width / 2
    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([pin_length / 2, -d])
    dtopin = np.array([pin_length / 2, +d])
    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    # inner square
    d = p.width / 2
    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([-pin_length / 2, -d])
    dtopin = np.array([-pin_length / 2, +d])
    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    x = (p0[0] + ptopin[0]) / 2
    y = (ptopin[1] + pbotin[1]) / 2
    if label_layer:
        component.add_label(
            text=str(p.name),
            position=(x, y),
            layer=label_layer,
        )


def add_pin_square(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Optional[Tuple[int, int]] = LAYER.PORT,
    port_margin: float = 0.0,
) -> None:
    """Add half out pin to a component.

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        label_layer: for the label
        port_margin: margin to port edge


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

    dbot = np.array([pin_length / 2, -d])
    dtop = np.array([pin_length / 2, d])
    dbotin = np.array([-pin_length / 2, -d])
    dtopin = np.array([-pin_length / 2, +d])

    p0 = p.position + _rotate(dbot, rot_mat)
    p1 = p.position + _rotate(dtop, rot_mat)
    ptopin = p.position + _rotate(dtopin, rot_mat)
    pbotin = p.position + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    if label_layer:
        component.add_label(
            text=str(p.name),
            position=p.midpoint,
            layer=label_layer,
        )


def add_outline(
    component: Component,
    reference: Optional[ComponentReference] = None,
    layer: Tuple[int, int] = LAYER.DEVREC,
    **kwargs,
) -> None:
    """Adds devices outline bounding box in layer.

    Args:
        component: where to add the markers
        reference: to read outline from
        layer: to add padding
        default: default padding
        top: North padding
        bottom
        right
        left
    """
    c = reference or component
    if hasattr(component, "parent"):
        component = component.parent
    points = get_padding_points(component=c, default=0, **kwargs)
    component.add_polygon(points, layer=layer)


def add_pins(
    component: Component,
    reference: Optional[ComponentReference] = None,
    function: Callable = add_pin_square_inside,
    port_type_to_layer: Dict[str, Tuple[int, int]] = PORT_TYPE_TO_LAYER,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports
        function:
        port_type_to_layer: dict mapping port types to marker layers for ports
        kwargs: add pins function settings

    """
    reference = reference or component
    for p in reference.ports.values():
        layer = port_type_to_layer[p.port_type]
        function(component=component, port=p, layer=layer, label_layer=layer, **kwargs)


def add_pins_triangle(**kwargs) -> None:
    return add_pins(function=add_pin_triangle, **kwargs)


def add_settings_label(
    component: Component,
    reference: ComponentReference,
    label_layer: Tuple[int, int] = LAYER.LABEL_SETTINGS,
    ignore: Optional[Iterable[str]] = None,
) -> None:
    """Add settings in label, ignores component.ignore keys.

    Args:
        componnent
        reference
        label_layer:
        ignore: fields to ignoreg

    """
    settings = reference.get_settings(ignore=ignore)
    settings_string = f"settings={json.dumps(settings, indent=2)}"
    if len(settings_string) > 1024:
        raise ValueError(f"label > 1024 characters: {settings_string}")
    component.add_label(
        position=reference.center, text=settings_string, layer=label_layer
    )


def add_instance_label(
    component: Component,
    reference: ComponentReference,
    instance_name: Optional[str] = None,
    layer: Tuple[int, int] = LAYER.LABEL_INSTANCE,
) -> None:
    """Adds label to a reference in a component."""

    instance_name = (
        instance_name
        or f"{reference.parent.name},{int(reference.x)},{int(reference.y)}"
    )
    x = gf.snap.snap_to_grid(reference.x)
    y = gf.snap.snap_to_grid(reference.y)

    component.add_label(
        text=instance_name,
        position=(x, y),
        layer=layer,
    )


def add_pins_and_outline(
    component: Component,
    reference: ComponentReference,
    add_outline_function: Optional[Callable] = add_outline,
    add_pins_function: Optional[Callable] = add_pins,
    add_settings_function: Optional[Callable] = add_settings_label,
    add_instance_label_function: Optional[Callable] = add_settings_label,
) -> None:
    """Add markers:
    - outline
    - pins for the ports
    - label for the name
    - label for the settings

    Args:
        component: where to add the markers
        reference
        add_outline_function
        add_pins_function: to add pins to ports
        add_settings_function: to add outline around the component
        add_instance_label_function: labels each instance

    """
    if add_outline_function:
        add_outline_function(component=component, reference=reference)
    if add_pins_function:
        add_pins_function(component=component, reference=reference)
    if add_settings_function:
        add_settings_function(component=component, reference=reference)
    if add_instance_label_function:
        add_instance_label_function(component=component, reference=reference)


def add_pins_to_references(
    component: Component,
    function: Callable = add_pins,
) -> None:
    """Add pins to Component references.

    Args:
        component: component
        function: function to add pins
    """
    references = component.references

    if references:
        for reference in references:
            add_pins_to_references(component=reference.parent, function=function)
    else:
        function(component=component)


@cell
def add_pins_container(
    component: Component,
    function: Callable = add_pin_square_double,
    port_type: str = "optical",
    layer: Layer = LAYER.PORT,
) -> Component:
    """Add pins to a Component and returns a new Component

    Args:
        component:
        function: function to add pin
        port_type: optical, dc
        layer: layer for port marker

    Returns:
        New component
    """

    component_new = gf.Component(f"{component.name}_pins")
    component_new.add_ref(component)

    for p in component.ports.values():
        if p.port_type == port_type:
            function(component=component_new, port=p, layer=layer, label_layer=layer)

    component_new.ports = component.ports.copy()
    return component_new


def test_add_pins() -> None:
    c1 = gf.components.straight_with_heater()
    c2 = add_pins_container(
        component=c1, function=add_pin_square, port_type="optical", layer=LAYER.PORT
    )
    c2 = add_pins_container(
        component=c2, function=add_pin_square, port_type="dc", layer=LAYER.PORTE
    )
    c2.show(show_ports=False)

    n_optical_expected = 2
    n_dc_expected = 2
    # polygons = 194

    port_layer_optical = PORT_TYPE_TO_LAYER["optical"]
    port_markers_optical = read_port_markers(c2, [port_layer_optical])
    n_optical = len(port_markers_optical.polygons)

    port_layer_dc = PORT_TYPE_TO_LAYER["dc"]
    port_markers_dc = read_port_markers(c2, [port_layer_dc])
    n_dc = len(port_markers_dc.polygons)

    # print(len(c1.get_polygons()))
    # print(len(c2.get_polygons()))
    print(n_optical)
    print(n_dc)

    # assert len(c1.get_polygons()) == polygons
    # assert len(c2.get_polygons()) == polygons + 41
    assert (
        n_optical == n_optical_expected
    ), f"{n_optical} optical pins different from {n_optical_expected}"
    assert (
        n_dc_expected == n_dc_expected
    ), f"{n_dc_expected} electrical pins different from {n_dc_expected}"


if __name__ == "__main__":
    # test_add_pins()

    # c = gf.components.straight()
    # add_pins(c, function=add_pin_square)
    # add_pins(c, function=add_pin_square_inside)
    # add_pins(c, function=add_pin_square_double)
    # c.show(show_ports=False)

    # cpl = [10, 20, 30]
    # cpg = [0.2, 0.3, 0.5]
    # dl0 = [10, 20]
    # c = gf.components.mzi_lattice(coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0)

    # c = gf.components.mzi()
    # add_pins_to_references(c)
    # c = add_pins(c, recursive=True)
    # c = add_pins(c, recursive=False)
    # c.show()

    # c = mmi1x2(width_mmi=5)
    # cc = add_grating_couplers(c, layer_label=gf.LAYER.LABEL)

    # c = gf.components.straight()
    # c = gf.components.crossing()
    # add_pins(c)

    # c = gf.components.bend_circular()
    # print(cc.name)
    # cc.show()

    c = gf.components.ring_single()
    add_pins_to_references(c)
    c.show()
