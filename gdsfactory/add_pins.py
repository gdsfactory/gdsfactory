"""Add_pin adds a Pin to a port, add_pins adds Pins to all ports.

- pins
- outline

Some functions modify a component without changing its name.
Make sure these functions are inside a new Component or called as a decorator
They without modifying the cell name
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import kfactory as kf
import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf

from gdsfactory.component import container
from gdsfactory.port import select_ports

if TYPE_CHECKING:
    from gdsfactory.component import Component, Instance
    from gdsfactory.port import Port

Layer = tuple[int, int]
Layers = tuple[Layer, ...]
LayerSpec = Layer | str | int | None
LayerSpecs = tuple[LayerSpec, ...]
nm = 1e-3


def _rotate(v: ndarray, m: ndarray) -> ndarray:
    return np.dot(m, v)


def add_bbox(
    component: Component,
    bbox_layer: LayerSpec = "DEVREC",
    top: float = 0,
    bottom: float = 0,
    left: float = 0,
    right: float = 0,
) -> Component:
    """Add bbox on outline.

    Args:
        component: component to add bbox.
        bbox_layer: bbox layer.
        top: padding.
        bottom: padding.
        left: padding.
        right: padding.
    """
    from gdsfactory.pdk import get_layer

    layer = get_layer(bbox_layer)
    bbox = component.dbbox()
    dxmin, dymin, dxmax, dymax = bbox.left, bbox.bottom, bbox.right, bbox.top
    points = [
        [dxmin - left, dymin - bottom],
        [dxmax + right, dymin - bottom],
        [dxmax + right, dymax + top],
        [dxmin - left, dymax + top],
    ]
    component.add_polygon(points, layer=layer)
    return component


def add_bbox_siepic(
    component: Component,
    bbox_layer: LayerSpec = "DEVREC",
    remove_layers: LayerSpecs = ("PORT", "PORTE"),
) -> Component:
    """Add bounding box device recognition layer.

    Args:
        component: to add bbox.
        bbox_layer: bounding box.
        remove_layers: remove other layers.
    """
    from gdsfactory.pdk import get_layer

    bbox_layer = get_layer(bbox_layer)
    remove_layers = remove_layers or []
    remove_layers = list(remove_layers) + [bbox_layer]
    remove_layers = [get_layer(layer) for layer in remove_layers]
    component = component.remove_layers(layers=remove_layers, recursive=False)

    if bbox_layer:
        component.add_padding(default=0, layers=(bbox_layer,))
    return component


def get_pin_triangle_polygon_tip(
    port: Port,
) -> tuple[list[float], tuple[float, float]]:
    """Returns triangle polygon and tip position."""
    p = port
    port_face = p.info.get("face", None)

    orientation = p.orientation

    if orientation is None:
        raise ValueError("Port {port.name!r} needs to have an orientation.")

    ca = np.cos(orientation * np.pi / 180)
    sa = np.sin(orientation * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])
    d = p.dwidth / 2

    dtip = np.array([d, 0])

    if port_face:
        dtop = port_face[0]
        dbot = port_face[-1]
    else:
        dbot = np.array([0, -d])
        dtop = np.array([0, d])

    p0 = p.dcenter + _rotate(dbot, rot_mat)
    p1 = p.dcenter + _rotate(dtop, rot_mat)
    port_face = [p0, p1]

    ptip = p.dcenter + _rotate(dtip, rot_mat)

    polygon = list(port_face) + [ptip]
    polygon = np.stack(polygon)
    return polygon, ptip


def add_pin_triangle(
    component: Component,
    port: Port,
    layer: LayerSpec = "PORT",
    layer_label: LayerSpec = "TEXT",
) -> None:
    """Add triangle pin with a right angle, pointing out of the port.

    Args:
        component: to add pin.
        port: Port.
        layer: for the pin marker.
        layer_label: for the label.
    """
    if port.orientation is not None:
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
    layer: LayerSpec = "PORT",
    layer_label: LayerSpec = "TEXT",
) -> None:
    """Add square pin towards the inside of the port.

    Args:
        component: to add pins.
        port: Port.
        pin_length: length of the pin marker for the port.
        layer: for the pin marker.
        layer_label: for the label.

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

    d = p.dwidth / 2

    dbot = np.array([0, -d])
    dtop = np.array([0, +d])
    dbotin = np.array([-pin_length, -d])
    dtopin = np.array([-pin_length, +d])

    p0 = p.dcenter + _rotate(dbot, rot_mat)
    p1 = p.dcenter + _rotate(dtop, rot_mat)
    ptopin = p.dcenter + _rotate(dtopin, rot_mat)
    pbotin = p.dcenter + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    if layer_label:
        component.add_label(
            text=str(p.name),
            position=p.dcenter,
            layer=layer_label,
        )


def add_pin_rectangle_double(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: LayerSpec = "PORT",
    layer_label: LayerSpec = "TEXT",
) -> None:
    """Add two square pins: one inside with label, one outside.

    Args:
        component: to add pins.
        port: Port.
        pin_length: length of the pin marker for the port.
        layer: for the pin marker.
        layer_label: for the label.

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
    d = p.dwidth / 2
    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([pin_length / 2, -d])
    dtopin = np.array([pin_length / 2, +d])
    p0 = p.dcenter + _rotate(dbot, rot_mat)
    p1 = p.dcenter + _rotate(dtop, rot_mat)
    ptopin = p.dcenter + _rotate(dtopin, rot_mat)
    pbotin = p.dcenter + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    # inner square
    d = p.dwidth / 2
    dbot = np.array([0, -d])
    dtop = np.array([0, d])
    dbotin = np.array([-pin_length / 2, -d])
    dtopin = np.array([-pin_length / 2, +d])
    p0 = p.dcenter + _rotate(dbot, rot_mat)
    p1 = p.dcenter + _rotate(dtop, rot_mat)
    ptopin = p.dcenter + _rotate(dtopin, rot_mat)
    pbotin = p.dcenter + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    dx = (p0[0] + ptopin[0]) / 2
    dy = (ptopin[1] + pbotin[1]) / 2
    if layer_label:
        component.add_label(
            text=str(p.name),
            position=(dx, dy),
            layer=layer_label,
        )


def add_pin_rectangle(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: LayerSpec = "PORT",
    layer_label: LayerSpec = "TEXT",
    port_margin: float = 0.0,
) -> None:
    """Add half out pin to a component.

    Args:
        component: to add pin.
        port: Port.
        pin_length: length of the pin marker for the port.
        layer: for the pin marker.
        layer_label: for the label.
        port_margin: margin to port edge.

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

    d = p.dwidth / 2 + port_margin

    dbot = np.array([pin_length / 2, -d])
    dtop = np.array([pin_length / 2, d])
    dbotin = np.array([-pin_length / 2, -d])
    dtopin = np.array([-pin_length / 2, +d])

    p0 = p.dcenter + _rotate(dbot, rot_mat)
    p1 = p.dcenter + _rotate(dtop, rot_mat)
    ptopin = p.dcenter + _rotate(dtopin, rot_mat)
    pbotin = p.dcenter + _rotate(dbotin, rot_mat)
    polygon = [p0, p1, ptopin, pbotin]
    component.add_polygon(polygon, layer=layer)

    if layer_label:
        component.add_label(
            text=str(p.name),
            position=p.dcenter,
            layer=layer_label,
        )


def add_pin_path(
    component: Component,
    port: Port,
    pin_length: float = 2 * nm,
    layer: LayerSpec = "PORT",
    layer_label: LayerSpec | None = None,
) -> None:
    """Add half out path pin to a component.

    This port type is compatible with SiEPIC pdk.

    Args:
        component: to add pin.
        port: Port.
        pin_length: length of the pin marker for the port.
        layer: for the pin marker.
        layer_label: optional layer label. Defaults to layer.

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
    from gdsfactory.pdk import get_layer

    layer_label = layer_label or layer
    p = port
    a = p.orientation
    ca = np.cos(a * np.pi / 180)
    sa = np.sin(a * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])

    d0 = np.array([-pin_length / 2, 0])
    d1 = np.array([+pin_length / 2, 0])

    p0 = p.dcenter + _rotate(d0, rot_mat)
    p1 = p.dcenter + _rotate(d1, rot_mat)

    points = [p0, p1]
    points = [kf.kdb.DPoint(p[0], p[1]) for p in points]
    layer = get_layer(layer)

    dpath = kf.kdb.DPath(
        points,
        p.dwidth,
    )
    component.add_label(text=str(p.name), position=p.dcenter, layer=layer_label)
    component.shapes(layer).insert(dpath)


def add_outline(
    component: Component,
    reference: Instance | None = None,
    layer: LayerSpec = "DEVREC",
    **kwargs,
) -> None:
    """Adds devices outline bounding box in layer.

    Args:
        component: where to add the markers.
        reference: to read outline from.
        layer: to add padding.
        kwargs: padding settings.

    Keyword Args:
        default: default padding.
        top: North padding.
        bottom: padding.
        right: padding.
        left: padding.
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
    layer: LayerSpec = "PORT",
    pin_length: float = 10 * nm,
    **kwargs,
) -> Component:
    """Add pins.

    Enables you to run SiEPIC verification tools:
    To Run verification install SiEPIC-tools KLayout package
    then hit V shortcut in KLayout to run verification

    - ensure no disconnected pins
    - netlist extraction

    Args:
        component: to add pins.
        function: to add pin.
        port_type: optical, electrical, ...
        layer: pin layer.
        pin_length: length of the pin marker for the port.
        kwargs: add pins function settings.
    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    for p in component.get_ports_list(port_type=port_type):
        function(
            component=component, port=p, layer=layer, pin_length=pin_length, **kwargs
        )

    return component


add_pins_siepic_optical = add_pins_siepic
add_pins_siepic_electrical = partial(
    add_pins_siepic, port_type="electrical", layer="PORTE"
)


def add_pins(
    component: Component,
    port_type: str | None = None,
    function: Callable = add_pin_rectangle_inside,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports to.
        port_type: Which port type do you want to add pins to. optical, electrical, ...  If None, it will add to all.
        layer: layer for the pin marker.
        function: to add each pin.
        kwargs: add pins function settings.
    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    # This should only select ports according to the port type
    ports = select_ports(
        ports=component.ports,
        port_type=port_type,
    )

    for port in ports:
        function(component=component, port=port, **kwargs)


add_pins_triangle = partial(add_pins, function=add_pin_triangle)
add_pins_center = partial(add_pins, function=add_pin_rectangle)
add_pin_inside1nm = partial(
    add_pin_rectangle_inside, pin_length=1 * nm, layer_label=None
)
add_pin_inside2um = partial(add_pin_rectangle_inside, pin_length=2, layer_label=None)
add_pins_inside1nm = partial(add_pins, function=add_pin_inside1nm)
add_pins_inside2um = partial(add_pins, function=add_pin_inside2um)


def add_settings_label(
    component: Component,
    reference: Instance | None = None,
    layer_label: LayerSpec = "LABEL_SETTINGS",
    with_yaml_format: bool = False,
) -> None:
    """Add settings in label.

    Args:
        component: to add pins.
        reference: Instance.
        layer_label: layer spec.
        with_yaml_format: add yaml format, False uses json.
    """
    from gdsfactory.pdk import get_layer

    layer_label = get_layer(layer_label)

    reference = reference or component
    settings_dict = OmegaConf.to_container(reference.info)
    settings_string = (
        OmegaConf.to_yaml(settings_dict)
        if with_yaml_format
        else f"settings={json.dumps(settings_dict)}"
    )
    if len(settings_string) > 1024:
        raise ValueError(f"label > 1024 characters: {settings_string}")
    component.add_label(
        position=reference.dcenter, text=settings_string, layer=layer_label
    )


def add_instance_label(
    component: Component,
    reference: Instance,
    instance_name: str | None = None,
    layer: LayerSpec = "LABEL_INSTANCE",
) -> None:
    """Adds label to a reference in a component.

    Args:
        component: to add instance label.
        reference: to add label.
        instance_name: label name.
        layer: layer for the label.

    """
    instance_name = (
        instance_name
        or f"{reference.parent.name},{int(reference.dx)},{int(reference.dy)}"
    )

    component.add_label(
        text=instance_name,
        position=reference.dcenter,
        layer=layer,
    )


def add_pins_and_outline(
    component: Component,
    reference: Instance | None = None,
    add_outline_function: Callable | None = add_outline,
    add_pins_function: Callable | None = add_pins,
    add_settings_function: Callable | None = add_settings_label,
    add_instance_label_function: Callable | None = add_settings_label,
) -> None:
    """Add pins component outline.

    Args:
        component: where to add the markers.
        reference: to add pins.
        add_outline_function: to add outline around the component.
        add_pins_function: to add pins to ports.
        add_settings_function: to add outline around the component.
        add_instance_label_function: labels each instance.
    """
    if add_outline_function:
        add_outline_function(component=component, reference=reference)
    if add_pins_function:
        add_pins_function(component=component, reference=reference)
    if add_settings_function:
        add_settings_function(component=component, reference=reference)
    if add_instance_label_function:
        add_instance_label_function(component=component, reference=reference)


add_pins_container = partial(container, function=add_pins)
add_pins_siepic_container = partial(container, function=add_pins_siepic)

if __name__ == "__main__":
    import gdsfactory as gf

    # c = test_add_pins()
    # c.show( )
    # c = gf.components.straight(length=2)
    # c.show(show_ports_suborts=True)
    # p1 = len(c1.get_polygons())
    # p2 = len(c2.get_polygons())
    # assert p2 == p1 + 2
    # c1 = gf.components.straight_heater_metal(length=2)
    c = gf.components.bend_euler()
    # c = add_pins_container(c)
    add_pins_triangle(c)
    # c = add_pins_container(c)
    # cc.show()
    # c.show(show_subports=True)
    # c.show( )
    c.show()
