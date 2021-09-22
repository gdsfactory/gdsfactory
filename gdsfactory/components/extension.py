import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.port import Port
from gdsfactory.types import ComponentOrFactory, Coordinate, Layer

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
    return (p0, p1, p2, p3)


def move_polar_rad_copy(pos: Coordinate, angle: float, length: float) -> ndarray:
    """Returns the points of a position (pos) with angle, by shifted by certain length

    Args:
        pos: position
        angle: in radians
        length: extension length

    """
    c = np.cos(angle)
    s = np.sin(angle)
    return pos + length * np.array([c, s])


@cell
def extend_port(port: Port, length: float, layer: Optional[Layer] = None) -> Component:
    """Returns a straight extension component out of a port.

    Args:
        port: port to extend
        length: extension length
        layer: for the straight section
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
    component: ComponentOrFactory = mmi1x2,
    port_names: Optional[Tuple[str, ...]] = None,
    length: float = 5.0,
    extension_factory: Optional[ComponentOrFactory] = None,
    port1: Optional[str] = None,
    port2: Optional[str] = None,
    port_type: str = "optical",
    centered: bool = False,
    **kwargs,
) -> Component:
    """Returns a new component with some ports extended
    it can accept an extension_factory or it defaults to the port
    width and layer of each extended port

    Args:
        component: component to extend ports
        port_names: specify an list of ports names, if None it extends all ports
        length: extension length
        extension_factory: function to extend ports (defaults to a straight)
        port1: input port name
        port2: output port name
        port_type: type of the ports to extend
        **kwargs
    """
    c = gf.Component()
    component = component() if callable(component) else component
    cref = c << component
    c.component = component

    if centered:
        cref.x = 0
        cref.y = 0

    ports_all = cref.get_ports_list()
    port_all_names = [p.name for p in ports_all]

    ports_to_extend = cref.get_ports_list(port_type=port_type, **kwargs)
    ports_to_extend_names = [p.name for p in ports_to_extend]
    port_names = port_names or ports_to_extend_names or port_all_names

    for port_name in port_names:
        if port_name not in port_all_names:
            warnings.warn(f"Port Name {port_name} not in {port_all_names}")

    for port in ports_all:
        port_name = port.name
        port = cref.ports[port_name]

        if port_name in port_names:

            def extension_factory_default(
                length=length, width=port.width, port_type=port.port_type
            ):
                return gf.components.hline(
                    length=length, width=width, layer=port.layer, port_type=port_type
                )

            if extension_factory:
                extension_component = extension_factory()
            else:
                extension_component = extension_factory_default(
                    length=length, width=port.width, port_type=port.port_type
                )
            port_labels = list(extension_component.ports.keys())
            port1 = port1 or port_labels[0]
            port2 = port2 or port_labels[-1]

            extension = c << extension_component
            extension.connect(port1, port)
            c.add_port(port_name, port=extension.ports[port2])
            c.absorb(extension)
        else:
            c.add_port(port_name, port=component.ports[port_name])
    return c


def test_extend_ports() -> Component:
    import gdsfactory.components as pc

    c = pc.cross(width=2)
    ce = extend_ports(component=c)
    assert len(c.ports) == len(ce.ports)
    p = len(ce.polygons)
    assert p == 4, p
    return ce


def test_extend_ports_selection() -> Component:
    import gdsfactory.components as pc

    c = pc.cross(width=2)
    ce = extend_ports(component=c, port_names=list(range(1, 4)))
    assert len(c.ports) == len(ce.ports)
    return ce


__all__ = ["extend_ports", "extend_port"]


if __name__ == "__main__":
    # c = extend_ports()
    # c = test_extend_ports_selection()
    # c = test_extend_ports()
    # c.show()

    # c = gf.c.bend_circular()
    # ce = extend_ports(component=c, port_names=list(c.ports.keys()) + ["hi"])
    # ce.show()

    wg_pin = gf.components.straight_pin(length=40)
    wg_pin.show()

    # c = pc.straight(layer=(3, 0))
    # print(ce)
    # print(len(ce.ports))
    # c = pc.straight()
    # ce = extend_ports(component=c)
    # ce.show()
