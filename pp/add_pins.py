"""add_pin adss a Pin to a port, add_pins adds Pins to all ports:

- pins
- outline

Do not use functions with name starting with underscore directly as they modify a component without changing its name.
Make sure underscore functions are inside a new Component as they modify the geometry of a component (add pins, labels, grating couplers ...) without modifying the cell name
You can use the @container decorator

"""
import json
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import pp
from pp.add_padding import get_padding_points
from pp.component import Component, ComponentReference
from pp.container import container
from pp.layers import LAYER, port_type2layer
from pp.port import Port, read_port_markers


def _rotate(v, m):
    return np.dot(m, v)


def _add_pin_triangle(
    component: Component,
    port: Port,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Tuple[int, int] = LAYER.TEXT,
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

    component.add_label(
        text=p.name, position=p.midpoint, layer=label_layer,
    )

    component.add_polygon(polygon, layer=layer)


def _add_pin_square_inside(
    component: Component,
    port: Port,
    pin_length: float = 0.1,
    layer: Tuple[int, int] = LAYER.PORT,
    label_layer: Tuple[int, int] = LAYER.TEXT,
) -> None:
    """Add square pin towards the inside of the port

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


def _add_pin_square(
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
            text=str(p.name), position=p.midpoint, layer=label_layer,
        )


def _add_outline(
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


def _add_pins(
    component: Component,
    reference: Optional[ComponentReference] = None,
    add_port_marker_function: Callable = _add_pin_square,
    port_type2layer: Dict[str, Tuple[int, int]] = port_type2layer,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports
        add_port_marker_function:
        port_type2layer: dict mapping port types to marker layers for ports

    """
    reference = reference or component
    for p in reference.ports.values():
        layer = port_type2layer[p.port_type]
        add_port_marker_function(
            component=component, port=p, layer=layer, label_layer=layer, **kwargs
        )


def _add_pins_triangle(**kwargs) -> None:
    return _add_pins(add_port_marker_function=_add_pin_triangle, **kwargs)


def _add_settings_label(
    component: Component,
    reference: ComponentReference,
    label_layer: Tuple[int, int] = LAYER.LABEL_SETTINGS,
) -> None:
    """Add settings in label, ignores component.ignore keys."""
    settings = reference.get_settings()
    settings_string = f"settings={json.dumps(settings, indent=2)}"
    component.add_label(
        position=reference.center, text=settings_string, layer=label_layer
    )


def _add_instance_label(
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
    component.add_label(
        text=instance_name,
        position=pp.drc.snap_to_1nm_grid((reference.x, reference.y)),
        layer=layer,
    )


def _add_pins_labels_and_outline(
    component: Component,
    reference: ComponentReference,
    add_outline_function: Optional[Callable] = _add_outline,
    add_pins_function: Optional[Callable] = _add_pins,
    add_settings_function: Optional[Callable] = _add_settings_label,
    add_instance_label_function: Optional[Callable] = _add_settings_label,
) -> None:
    """Add markers:
    - outline
    - pins for the ports
    - label for the name
    - label for the settings

    Args:
        component: where to add the markers
        pins_function: function to add pins to ports
        add_outline_function: function to add outline around the device

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
    references: Optional[List[ComponentReference]] = None,
    function: Callable = _add_pins_labels_and_outline,
) -> None:
    """Add pins to a Component.

    Args:
        component: component
        references: list of references, taken from component by default
        function: function to add pins
    """
    references = references or component.references
    for reference in references:
        function(component=component, reference=reference)


@container
def add_pins(
    component: Component,
    function: Callable = _add_pins_labels_and_outline,
    recursive: bool = False,
) -> Component:
    """Add pins to a Component and returns a container

    Args:
        component:
        function: function to add pins
        recursive: goes down the hierarchy

    Returns:
        New component
    """

    component_new = pp.Component(f"{component.name}_pins")
    reference = component_new << component
    function(component=component_new, reference=reference)

    if recursive:
        for reference in component.references:
            function(component=component_new, reference=reference)

    return component_new


def test_add_pins():
    c1 = pp.c.mzi2x2(with_elec_connections=True)
    c2 = add_pins(c1, recursive=False)

    n_optical_expected = 4
    n_dc_expected = 3
    # polygons = 194

    port_layer_optical = port_type2layer["optical"]
    port_markers_optical = read_port_markers(c2, [port_layer_optical])
    n_optical = len(port_markers_optical.polygons)

    port_layer_dc = port_type2layer["dc"]
    port_markers_dc = read_port_markers(c2, [port_layer_dc])
    n_dc = len(port_markers_dc.polygons)

    # print(len(c1.get_polygons()))
    # print(len(c2.get_polygons()))
    print(n_optical)
    print(n_dc)

    # assert len(c1.get_polygons()) == polygons
    # assert len(c2.get_polygons()) == polygons + 41
    assert n_optical == n_optical_expected
    assert n_dc_expected == n_dc_expected


def test_add_pins_recursive():
    c1 = pp.c.mzi2x2(with_elec_connections=True)
    c2 = add_pins(c1, recursive=True)
    pp.show(c2)

    n_optical_expected = 22
    n_dc_expected = 11

    port_layer_optical = port_type2layer["optical"]
    port_markers_optical = read_port_markers(c2, [port_layer_optical])
    n_optical = len(port_markers_optical.polygons)

    port_layer_dc = port_type2layer["dc"]
    port_markers_dc = read_port_markers(c2, [port_layer_dc])
    n_dc = len(port_markers_dc.polygons)

    print(n_optical)
    print(n_dc)

    assert n_optical == n_optical_expected
    assert n_dc_expected == n_dc_expected


if __name__ == "__main__":
    cpl = [10, 20, 30]
    cpg = [0.2, 0.3, 0.5]
    dl0 = [10, 20]
    # c = pp.c.mzi_lattice(coupler_lengths=cpl, coupler_gaps=cpg, delta_lengths=dl0)

    # c = pp.c.mzi()
    # add_pins_to_references(c)
    # c = add_pins(c, recursive=True)
    # c = add_pins(c, recursive=False)
    # pp.show(c)

    # c = pp.c.mzi2x2(with_elec_connections=True)
    # cc = add_pins(c)
    # pp.show(cc)

    # test_add_pins()
    test_add_pins_recursive()

    # from pp.components import mmi1x2
    # from pp.components import bend_circular
    # from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    # cc = add_grating_couplers(c, layer_label=pp.LAYER.LABEL)

    # c = pp.c.waveguide()
    # c = pp.c.crossing()
    # add_pins(c)

    # c = pp.c.bend_circular()
    # cc = pp.containerize(component=c, function=add_outline)
    # print(cc.name)
    # pp.show(cc)
