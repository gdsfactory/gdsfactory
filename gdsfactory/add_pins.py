"""Add_pin adds a Pin to a port, add_pins adds Pins to all ports.

- pins
- outline

Some functions modify a component without changing its name.
Make sure these functions are inside a new Component or called as a decorator
They without modifying the cell name
"""

from __future__ import annotations

import json
import warnings
from functools import partial
from typing import Any, Protocol, cast

import kfactory as kf
import numpy as np
import numpy.typing as npt
import yaml

import gdsfactory as gf
from gdsfactory import typings
from gdsfactory.component import Component, ComponentReference, container
from gdsfactory.config import CONF
from gdsfactory.port import select_ports
from gdsfactory.serialization import convert_tuples_to_lists

nm = 1e-3


def _rotate(
    vector: npt.NDArray[np.floating[Any]],
    rotation_matrix: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Rotate a vector by a rotation matrix."""
    return rotation_matrix @ vector


def add_bbox(
    component: Component,
    bbox_layer: typings.LayerSpec = "DEVREC",
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
        (dxmin - left, dymin - bottom),
        (dxmax + right, dymin - bottom),
        (dxmax + right, dymax + top),
        (dxmin - left, dymax + top),
    ]
    component.add_polygon(points, layer=layer)
    return component


def add_bbox_siepic(
    component: Component,
    bbox_layer: typings.LayerSpec = "DEVREC",
    remove_layers: typings.LayerSpecs = ("PORT", "PORTE"),
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
    port: typings.Port,
) -> tuple[npt.NDArray[np.floating[Any]], tuple[float, float]]:
    """Returns triangle polygon and tip position."""
    p = port
    port_face = p.info.get("face", None)

    orientation = p.orientation

    ca = np.cos(orientation * np.pi / 180)
    sa = np.sin(orientation * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])
    d = p.width / 2

    dtip = np.array([d, 0])

    if port_face:
        dtop = port_face[0]
        dbot = port_face[-1]
    else:
        dbot = np.array([0, -d])
        dtop = np.array([0, d])

    p0 = p.center + _rotate(dbot, rot_mat)
    p1 = p.center + _rotate(dtop, rot_mat)
    port_face = [p0, p1]

    ptip = cast(
        tuple[float, float], tuple(map(float, p.center + _rotate(dtip, rot_mat)))
    )

    polygon = list(port_face) + [ptip]
    polygon_stacked = np.stack(polygon)
    return polygon_stacked, ptip


def add_pin_triangle(
    component: Component,
    port: typings.Port,
    layer: typings.LayerSpec = "PORT",
    layer_label: typings.LayerSpec | None = "TEXT",
) -> None:
    """Add triangle pin with a right angle, pointing out of the port.

    Args:
        component: to add pin.
        port: Port.
        layer: for the pin marker.
        layer_label: for the label.
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
    port: typings.Port,
    pin_length: float = 0.1,
    layer: typings.LayerSpec = "PORT",
    layer_label: typings.LayerSpec | None = "TEXT",
) -> None:
    """Add square pin towards the inside of the port.

    Args:
        component: to add pins.
        port: Port.
        pin_length: length of the pin marker for the port.
        layer: layer to place the pin rectangle on.
        layer_label: layer to place the text label on.

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
    if layer:
        p = port
        poly = gf.kdb.DPolygon(
            gf.kdb.DBox(-pin_length, -p.width / 2, 0, p.width / 2)
        ).transform(p.dcplx_trans)
        component.shapes(gf.get_layer(layer)).insert(poly)

    if layer_label:
        assert port.name is not None
        component.add_label(
            text=port.name,
            position=port.center,
            layer=layer_label,
        )


def add_pin_rectangle(
    component: Component,
    port: typings.Port,
    pin_length: float = 0.1,
    layer: typings.LayerSpec | None = "PORT",
    layer_label: typings.LayerSpec | None = "TEXT",
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
    if layer:
        width = port.width + port_margin
        poly = gf.kdb.DPolygon(
            gf.kdb.DBox(-pin_length / 2, -width / 2, +pin_length / 2, width / 2)
        ).transform(port.dcplx_trans)
        component.shapes(gf.get_layer(layer)).insert(poly)

    if layer_label:
        component.add_label(
            text=str(port.name),
            position=port.center,
            layer=layer_label,
        )


class AddPinPathFunction(Protocol):
    def __call__(
        self,
        component: Component,
        port: typings.Port,
        pin_length: float = ...,
        layer: typings.LayerSpec = ...,
        layer_label: typings.LayerSpec | None = ...,
    ) -> None: ...


def add_pin_path(
    component: Component,
    port: typings.Port,
    pin_length: float = 2 * nm,
    layer: typings.LayerSpec = "PORT",
    layer_label: typings.LayerSpec | None = None,
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

    p0 = p.center + _rotate(d0, rot_mat)
    p1 = p.center + _rotate(d1, rot_mat)

    points = [p0, p1]
    dpoints = [kf.kdb.DPoint(p[0], p[1]) for p in points]
    layer = get_layer(layer)

    dpath = kf.kdb.DPath(
        dpoints,
        p.width,
    )
    component.add_label(text=str(p.name), position=p.center, layer=layer_label)
    component.shapes(layer).insert(dpath)


def add_outline(
    component: Component,
    reference: ComponentReference | None = None,
    layer: typings.LayerSpec = "DEVREC",
    **kwargs: Any,
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
    function: AddPinPathFunction = add_pin_path,
    port_type: str = "optical",
    layer: typings.LayerSpec = "PORT",
    pin_length: float = 10 * nm,
    **kwargs: Any,
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


class AddPinFunction(Protocol):
    def __call__(
        self,
        component: Component,
        port: typings.Port,
        **kwargs: Any,
    ) -> Any: ...


def add_pins(
    component: Component,
    port_type: str | None = None,
    function: AddPinFunction = add_pin_rectangle_inside,  # type: ignore[assignment]
    **kwargs: Any,
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
        function(component, port, **kwargs)


add_pins_triangle = partial(add_pins, function=add_pin_triangle)  # type: ignore[arg-type]
add_pins_center = partial(add_pins, function=add_pin_rectangle)  # type: ignore[arg-type]
add_pin_inside1nm = partial(
    add_pin_rectangle_inside, pin_length=1 * nm, layer_label=None
)
add_pin_inside2um = partial(add_pin_rectangle_inside, pin_length=2, layer_label=None)
add_pins_inside1nm = partial(add_pins, function=add_pin_inside1nm)
add_pins_inside2um = partial(add_pins, function=add_pin_inside2um)


def add_settings_label(
    component: Component,
    reference: ComponentReference | None = None,
    layer_label: typings.LayerSpec = "LABEL_SETTINGS",
    with_yaml_format: bool = False,
) -> None:
    """Add settings in label.

    Args:
        component: to add pins.
        reference: ComponentReference.
        layer_label: layer spec.
        with_yaml_format: add yaml format, False uses json.
    """
    from gdsfactory.pdk import get_layer

    layer_label = get_layer(layer_label)

    reference_or_component = reference or component
    info = (
        reference_or_component.cell.info
        if hasattr(reference_or_component, "cell")
        else reference_or_component.info
    )
    settings_dict = dict(info)
    settings_string = (
        yaml.dump(convert_tuples_to_lists(settings_dict))
        if with_yaml_format
        else f"settings={json.dumps(settings_dict)}"
    )
    if len(settings_string) > 1024:
        raise ValueError(f"label > 1024 characters: {settings_string}")
    component.add_label(
        position=reference_or_component.center, text=settings_string, layer=layer_label
    )


def add_instance_label(
    component: Component,
    reference: ComponentReference,
    layer: typings.LayerSpec | None = None,
    instance_name: str | None = None,
) -> None:
    """Adds label to a reference in a component.

    Args:
        component: to add instance label.
        reference: to add label.
        layer: layer for the label.
        instance_name: label name.

    """
    try:
        layer = layer or gf.get_layer("LABEL_INSTANCE")
    except ValueError:
        warnings.warn("Layer LABEL_INSTANCE not found in PDK.layers, using (1, 0)")
        layer = (1, 0)
    instance_name = (
        instance_name or f"{reference.cell.name},{int(reference.x)},{int(reference.y)}"
    )

    layer = layer or CONF.layer_label

    component.add_label(
        text=instance_name,
        position=reference.center,
        layer=layer,
    )


class AddInstanceLabelFunction(Protocol):
    def __call__(
        self, component: Component, reference: ComponentReference | None = None
    ) -> None: ...


class AddPinsFunction(Protocol):
    def __call__(
        self, component: Component, reference: ComponentReference | None = None
    ) -> None: ...


def add_pins_and_outline(
    component: Component,
    reference: ComponentReference | None = None,
    add_outline_function: AddInstanceLabelFunction | None = add_outline,
    add_pins_function: AddPinsFunction | None = add_pins,  # type: ignore[assignment]
    add_settings_function: AddInstanceLabelFunction | None = add_settings_label,
    add_instance_label_function: AddInstanceLabelFunction | None = add_settings_label,
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
        add_pins_function(component, reference)
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
    # c = gf.components.bend_euler()
    # c = add_pins_container(c)
    # add_pins_triangle(c)
    # c = add_pins_container(c)
    # cc.show()
    # c.show(show_subports=True)
    # c.show( )

    c = gf.Component()
    ref = c << gf.components.straight()
    c.add_ports(ref.ports)

    add_pin_rectangle(c, port=c.ports[0], port_margin=1)
    c.show()
