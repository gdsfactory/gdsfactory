from __future__ import annotations

import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.cross_section import cross_section as cross_section_function
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, Coordinate, CrossSectionSpec, Layer

DEG2RAD = np.pi / 180


def line(
    p_start: Union[Port, Coordinate],
    p_end: Union[Port, Coordinate],
    width: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if isinstance(p_start, gf.Port):
        width = p_start.width
        p_start = p_start.center

    if isinstance(p_end, gf.Port):
        p_end = p_end.center

    w = width
    angle = np.arctan2(p_end[1] - p_start[1], p_end[0] - p_start[0])
    a = np.pi / 2
    p0 = move_polar_rad_copy(p_start, angle + a, w / 2)
    p1 = move_polar_rad_copy(p_start, angle - a, w / 2)
    p2 = move_polar_rad_copy(p_end, angle - a, w / 2)
    p3 = move_polar_rad_copy(p_end, angle + a, w / 2)
    return p0, p1, p2, p3


def move_polar_rad_copy(pos: Coordinate, angle: float, length: float) -> ndarray:
    """Returns the points of a position (pos) with angle, shifted by length.

    Args:
        pos: position.
        angle: in radians.
        length: extension length in um.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return pos + length * np.array([c, s])


@cell
def extend_port(port: Port, length: float, layer: Optional[Layer] = None) -> Component:
    """Returns a straight extension component out of a port.

    Args:
        port: port to extend.
        length: extension length in um.
        layer: for the straight section.
    """
    c = Component()
    layer = layer or port.layer

    # Generate a port extension
    p_start = port.center
    angle = port.orientation
    p_end = move_polar_rad_copy(p_start, angle * DEG2RAD, length)
    w = port.width

    _line = line(p_start, p_end, w)

    c.add_polygon(_line, layer=layer)
    c.add_port(name="original", port=port)

    port_settings = port.to_dict()
    port_settings.update(center=p_end)
    c.add_port(**port_settings)

    return c


@gf.cell
def extend_ports(
    component: ComponentSpec = mmi1x2,
    port_names: Optional[Tuple[str, ...]] = None,
    length: float = 5.0,
    extension: Optional[ComponentSpec] = None,
    port1: Optional[str] = None,
    port2: Optional[str] = None,
    port_type: str = "optical",
    centered: bool = False,
    cross_section: Optional[CrossSectionSpec] = None,
    extension_port_names: Optional[List[str]] = None,
    **kwargs,
) -> Component:
    """Returns a new component with some ports extended.

    You can define extension Spec
    defaults to port cross_section of each port to extend.

    Args:
        component: component to extend ports.
        port_names: list of ports names to extend, if None it extends all ports.
        length: extension length.
        extension: function to extend ports (defaults to a straight).
        port1: extension input port name.
        port2: extension output port name.
        port_type: type of the ports to extend.
        centered: if True centers rectangle at (0, 0).
        cross_section: extension cross_section, defaults to port cross_section
            if port has no cross_section it creates one using width and layer.
        extension_port_names: extension port names add to the new component.

    Keyword Args:
        layer: port GDS layer.
        prefix: port name prefix.
        orientation: in degrees.
        width: port width.
        layers_excluded: List of layers to exclude.
        port_type: optical, electrical, ....
        clockwise: if True, sort ports clockwise, False: counter-clockwise.
    """
    c = gf.Component()
    component = gf.get_component(component)
    cref = c << component

    if centered:
        cref.x = 0
        cref.y = 0

    ports_all = cref.get_ports_list()
    port_names_all = [p.name for p in ports_all]

    ports_to_extend = cref.get_ports_list(port_type=port_type, **kwargs)
    ports_to_extend_names = [p.name for p in ports_to_extend]
    ports_to_extend_names = port_names or ports_to_extend_names

    for port_name in ports_to_extend_names:
        if port_name not in port_names_all:
            warnings.warn(
                f"Port Name {port_name!r} not in {port_names_all}", stacklevel=3
            )

    for port in ports_all:
        port_name = port.name
        port = cref.ports[port_name]

        if port_name in ports_to_extend_names:
            if extension:
                extension_component = gf.get_component(extension)
            else:
                cross_section_extension = (
                    cross_section
                    or port.cross_section
                    or cross_section_function(layer=port.layer, width=port.width)
                )

                if cross_section_extension is None:
                    raise ValueError("cross_section=None for extend_ports")

                extension_component = gf.components.straight(
                    length=length,
                    cross_section=cross_section_extension,
                )
            port_labels = list(extension_component.ports.keys())
            port1 = port1 or port_labels[0]
            port2 = port2 or port_labels[-1]

            extension_ref = c << extension_component
            extension_ref.connect(port1, port)
            c.add_port(port_name, port=extension_ref.ports[port2])
            extension_port_names = extension_port_names or []
            [
                c.add_port(name, port=extension_ref.ports[name])
                for name in extension_port_names
            ]
        else:
            c.add_port(port_name, port=component.ports[port_name])

    c.copy_child_info(component)
    return c


def test_extend_ports() -> None:
    import gdsfactory.components as pc

    width = 0.5
    xs_strip = partial(
        gf.cross_section.strip,
        width=width,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )

    c = pc.straight(cross_section=xs_strip)

    c1 = extend_ports(
        component=c,
        cross_section=xs_strip,
    )
    assert len(c.ports) == len(c1.ports)
    p = len(c1.polygons)
    assert p == 0, p
    assert len(c1.references) == 3, len(c1.references)

    c2 = extend_ports(component=c, cross_section=xs_strip, port_names=("o1",))
    p = len(c2.polygons)
    assert p == 0, p
    assert len(c2.references) == 2, len(c2.references)

    c3 = extend_ports(component=c, cross_section=xs_strip)
    p = len(c3.polygons)
    assert p == 0, p

    c4 = extend_ports(component=c, port_type="electrical")
    p = len(c4.polygons)
    assert p == 0, p
    assert len(c4.references) == 1, len(c4.references)


__all__ = ["extend_ports", "extend_port"]


if __name__ == "__main__":
    c0 = gf.c.straight()
    p0 = c0["o1"]
    c = extend_port(p0, length=100)
    c.show()
