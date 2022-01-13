"""add_pin adss a Pin to a port, add_pins adds Pins to all ports:

- pins
- outline

Some functions modify a component without changing its name.
Make sure these functions are inside a new Component or called as a decorator
They without modifying the cell name

"""
import json
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component, ComponentReference
from gdsfactory.port import Port
from gdsfactory.tech import LAYER


def _rotate(v: ndarray, m: ndarray) -> ndarray:
    return np.dot(m, v)


def get_pin_triangle_polygon_tip(port: Port) -> Tuple[List[float], Tuple[float, float]]:
    """Returns triangle polygon and tip position"""
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
    polygon = np.stack(polygon)
    return polygon, ptip


def add_pin_triangle(
    component: Component,
    port: Port,
    layer: Tuple[int, int] = LAYER.PORT,
    layer_label: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add triangle pin with a right angle, pointing out of the port

    Args:
        component:
        port: Port
        layer: for the pin marker
        layer_label: for the label
    """
    polygon, ptip = get_pin_triangle_polygon_tip(port=port)
    component.add_polygon(polygon, layer=layer)

    if layer_label:
        component.add_label(
            text=str(port.name),
            position=ptip,
            layer=layer_label,
        )


def add_pin_square_inside(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    layer_label: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add square pin towards the inside of the port

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        layer_label: for the label

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
    if layer_label:
        component.add_label(
            text=str(p.name),
            position=p.midpoint,
            layer=layer_label,
        )


def add_pin_square_double(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    layer_label: Optional[Tuple[int, int]] = LAYER.TEXT,
) -> None:
    """Add two square pins: one inside with label, one outside

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        layer_label: for the label

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
    if layer_label:
        component.add_label(
            text=str(p.name),
            position=(x, y),
            layer=layer_label,
        )


def add_pin_square(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    layer_label: Optional[Tuple[int, int]] = LAYER.PORT,
    port_margin: float = 0.0,
) -> None:
    """Add half out pin to a component.

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker
        layer_label: for the label
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

    if layer_label:
        component.add_label(
            text=str(p.name),
            position=p.midpoint,
            layer=layer_label,
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
    select_ports: Optional[Callable] = None,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Be careful with this function as it modifies the component

    Args:
        component: to add ports to
        reference:
        function: to add each pin
        select_ports: function to select_ports
        kwargs: add pins function settings

    """
    reference = reference or component
    ports = (
        select_ports(reference.ports).values()
        if select_ports
        else reference.get_ports_list()
    )
    for port in ports:
        function(component=component, port=port, **kwargs)


add_pins_triangle = gf.partial(add_pins, function=add_pin_triangle)


@gf.cell
def add_pins_container(
    component: Component,
    add_pins_function=add_pins_triangle,
) -> Component:
    c = Component()
    ref = c << component
    add_pins_function(component=c, reference=ref)
    c.ref = ref
    return c


def add_settings_label(
    component: Component,
    reference: ComponentReference,
    layer_label: Tuple[int, int] = LAYER.LABEL_SETTINGS,
) -> None:
    """Add settings in label

    Args:
        componnent
        reference
        layer_label:

    """
    settings_dict = OmegaConf.to_container(reference.settings.full)
    settings_string = f"settings={json.dumps(settings_dict)}"
    print(settings_string)
    if len(settings_string) > 1024:
        raise ValueError(f"label > 1024 characters: {settings_string}")
    component.add_label(
        position=reference.center, text=settings_string, layer=layer_label
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


@gf.cell
def add_pins_to_references(
    component: Component,
    function: Callable = add_pins_triangle,
) -> Component:
    """Add pins to Component references.

    Args:
        component: component to add pins to
        references:
        function: function to add pins
    """
    c = Component()
    c.add_ref(component)

    references = component.references

    for reference in references:
        function(component=c, reference=reference)

    return c


def test_add_pins() -> None:
    c1 = gf.c.straight(length=2)
    c2 = add_pins_container(component=c1)

    p1 = len(c1.get_polygons())
    p2 = len(c2.get_polygons())

    assert p2 == p1 + 2
    assert len(c2.polygons) == 2


if __name__ == "__main__":
    # c = test_add_pins()
    # c.show()

    # c = gf.c.straight(length=2)
    # c = add_pins_container(component=c)

    # c.show(show_ports_suborts=True)

    # c2 = add_pins_container(component=c1)

    # p1 = len(c1.get_polygons())
    # p2 = len(c2.get_polygons())
    # assert p2 == p1 + 2

    # c1 = gf.c.straight_heater_metal(length=2)
    # c2 = add_pins_to_references(c1)
    c = gf.c.ring_single()
    # cc = add_pins_to_references(component=c)
    # cc.show(show_ports=False)
    c.show(show_subports=True)
    c.show(show_ports=True)
