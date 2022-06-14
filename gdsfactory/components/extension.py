import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.cross_section import cross_section as cross_section_function
from gdsfactory.port import Port
from gdsfactory.types import ComponentSpec, Coordinate, CrossSectionSpec, Layer

DEG2RAD = np.pi / 180


def line(
    p_start: Union[Port, Coordinate],
    p_end: Union[Port, Coordinate],
    width: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if isinstance(p_start, gf.Port):
        width = p_start.width
        p_start = p_start.midpoint

    if isinstance(p_end, gf.Port):
        p_end = p_end.midpoint

    w = width
    angle = np.arctan2(p_end[1] - p_start[1], p_end[0] - p_start[0])
    a = np.pi / 2
    p0 = move_polar_rad_copy(p_start, angle + a, w / 2)
    p1 = move_polar_rad_copy(p_start, angle - a, w / 2)
    p2 = move_polar_rad_copy(p_end, angle - a, w / 2)
    p3 = move_polar_rad_copy(p_end, angle + a, w / 2)
    return p0, p1, p2, p3


def move_polar_rad_copy(pos: Coordinate, angle: float, length: float) -> ndarray:
    """Returns the points of a position (pos) with angle, by shifted by certain length

    Args:
        pos: position.
        angle: in radians.
        length: extension length.

    """
    c = np.cos(angle)
    s = np.sin(angle)
    return pos + length * np.array([c, s])


@cell
def extend_port(port: Port, length: float, layer: Optional[Layer] = None) -> Component:
    """Returns a straight extension component out of a port.

    Args:
        port: port to extend.
        length: extension length.
        layer: for the straight section.
    """
    c = Component()
    layer = layer or port.layer

    # Generate a port extension
    p_start = port.midpoint
    angle = port.angle
    p_end = move_polar_rad_copy(p_start, angle * DEG2RAD, length)
    w = port.width

    _line = line(p_start, p_end, w)

    c.add_polygon(_line, layer=layer)
    c.add_port(name="original", port=port)

    port_settings = port.settings.copy()
    port_settings.update(midpoint=p_end)
    c.add_port(**port_settings)

    return c


@gf.cell
def extend_ports(
    component: ComponentSpec = mmi1x2,
    port_names: Optional[Tuple[str, ...]] = None,
    length: float = 5.0,
    extension_factory: Optional[ComponentSpec] = None,
    port1: Optional[str] = None,
    port2: Optional[str] = None,
    port_type: str = "optical",
    centered: bool = False,
    cross_section: Optional[CrossSectionSpec] = None,
    **kwargs,
) -> Component:
    """Returns a new component with some ports extended
    you can define an extension_factory
    otherwise, defaults to the port cross_section of each port to extend

    Args:
        component: component to extend ports.
        port_names: list of ports names to extend, if None it extends all ports.
        length: extension length.
        extension_factory: function to extend ports (defaults to a straight).
        port1: input port name.
        port2: output port name.
        port_type: type of the ports to extend.
        centered: if True centers rectangle at (0, 0).
        cross_section: extension cross_section, defaults to port cross_section
            if port has no cross_section it creates one using width and layer.

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
    c.component = component

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
            warnings.warn(f"Port Name {port_name} not in {port_names_all}")

    for port in ports_all:
        port_name = port.name
        port = cref.ports[port_name]

        if port_name in ports_to_extend_names:
            if extension_factory:
                extension_component = gf.get_component(extension_factory)
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

            extension = c << extension_component
            extension.connect(port1, port)
            c.add_port(port_name, port=extension.ports[port2])
        else:
            c.add_port(port_name, port=component.ports[port_name])

    c.copy_child_info(component)
    return c


def test_extend_ports() -> Component:
    import gdsfactory.components as pc

    width = 0.5
    xs_strip = gf.partial(
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

    return c4


__all__ = ["extend_ports", "extend_port"]


if __name__ == "__main__":
    # c = extend_ports()
    # c = extend_ports(gf.components.mzi_phase_shifter_top_heater_metal)
    c = test_extend_ports()

    # width = 0.5
    # xs_strip = gf.partial(
    #     gf.cross_section.strip,
    #     width=width,
    #     cladding_layers=None,
    #     add_pins=None,
    #     add_bbox=None,
    # )

    # import gdsfactory.components as pc
    # c = pc.straight(cross_section=xs_strip)
    # c4 = extend_ports(component=c, port_type='electrical')
    # c4.show()
