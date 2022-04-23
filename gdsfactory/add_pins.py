"""add_pin adss a Pin to a port, add_pins adds Pins to all ports:

- pins
- outline

Some functions modify a component without changing its name.
Make sure these functions are inside a new Component or called as a decorator
They without modifying the cell name

"""
import json
from functools import partial
from typing import Callable, List, Optional, Tuple

import gdspy
import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf
from phidl.device_layout import Device as Component
from phidl.device_layout import DeviceReference as ComponentReference

from gdsfactory.port import Port
from gdsfactory.snap import snap_to_grid
from gdsfactory.tech import LAYER

Layer = Tuple[int, int]
nm = 1e-3


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


def add_pin_rectangle_inside(
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


def add_pin_rectangle_double(
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


def add_pin_rectangle(
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


def add_pin_path(
    component: Component,
    port: Port,
    pin_length: float = 2 * nm,
    layer: Tuple[int, int] = LAYER.PORT,
    layer_label: Optional[Tuple[int, int]] = None,
) -> None:
    """Add half out path pin to a component.

    This port type is compatible with SiEPIC pdk.

    Args:
        component:
        port: Port
        pin_length: length of the pin marker for the port
        layer: for the pin marker.
        layer_label: for the label. Defaults to layer.


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
    layer_label = layer_label or layer
    p = port
    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d0 = np.array([-pin_length / 2, 0])
    d1 = np.array([+pin_length / 2, 0])

    p0 = p.position + _rotate(d0, rot_mat)
    p1 = p.position + _rotate(d1, rot_mat)

    points = [p0, p1]
    path = gdspy.FlexPath(
        points=points, width=p.width, layer=layer[0], datatype=layer[1], gdsii_path=True
    )
    component.add(path)

    component.add_label(
        text=str(p.name), position=p.midpoint, layer=layer_label, anchor="sw"
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
    from gdsfactory.add_padding import get_padding_points

    c = reference or component
    if hasattr(component, "parent"):
        component = component.parent
    points = get_padding_points(component=c, default=0, **kwargs)
    component.add_polygon(points, layer=layer)


def add_pins_siepic(
    component: Component,
    function: Callable = add_pin_path,
    port_type: str = "optical",
    layer_pin: Layer = LAYER.PORT,
    pin_length: float = 10 * nm,
) -> Component:
    """Add pins
    Enables you to run SiEPIC verification tools:
    To Run verification install SiEPIC-tools klayout package
    then hit V shortcut in klayout to run verification

    - ensure no disconnected pins
    - netlist extraction

    Args:
        component: to add pins.
        function: to add pin.
        port_type: optical, electrical, ...
        layer_pin: pin layer.
        pin_length: length of the pin marker for the port


    """

    for p in component.get_ports_list(port_type=port_type):
        function(component=component, port=p, layer=layer_pin, pin_length=pin_length)

    return component


add_pins_siepic_optical = add_pins_siepic
add_pins_siepic_electrical = partial(
    add_pins_siepic, port_type="electrical", layer_pin=LAYER.PORTE
)


def add_bbox_siepic(
    component: Component,
    bbox_layer: Optional[Layer] = (68, 0),
    padding: float = 0,
) -> Component:
    """Add bounding box device recognition layer."""
    if bbox_layer:
        component.add_padding(default=padding, layers=(bbox_layer,))
    return component


def add_pins_bbox_siepic(
    component: Component,
    function: Callable = add_pin_path,
    port_type: str = "optical",
    layer_pin: Layer = LAYER.PORT,
    pin_length: float = 10 * nm,
    bbox_layer: Layer = (68, 0),
    padding: float = 0,
) -> Component:
    """Add bounding box device recognition layer."""
    component = component.copy()
    component.remove_layers(layers=(layer_pin, bbox_layer))
    component.add_padding(default=padding, layers=(bbox_layer,))

    component = add_pins_siepic(
        component=component,
        function=function,
        port_type=port_type,
        layer_pin=layer_pin,
        pin_length=pin_length,
    )
    return component


def add_pins(
    component: Component,
    reference: Optional[ComponentReference] = None,
    function: Callable = add_pin_rectangle_inside,
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


add_pins_triangle = partial(add_pins, function=add_pin_triangle)


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
    x = snap_to_grid(reference.x)
    y = snap_to_grid(reference.y)

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


if __name__ == "__main__":
    import gdsfactory as gf

    # c = test_add_pins()
    # c.show()
    # c = gf.components.straight(length=2)
    # c.show(show_ports_suborts=True)
    # p1 = len(c1.get_polygons())
    # p2 = len(c2.get_polygons())
    # assert p2 == p1 + 2
    # c1 = gf.components.straight_heater_metal(length=2)
    c = gf.components.ring_single()
    # cc.show(show_ports=False)
    c.show(show_subports=True)
    c.show(show_ports=True)
