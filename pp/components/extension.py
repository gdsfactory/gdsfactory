from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.mmi1x2 import mmi1x2
from pp.port import Port
from pp.types import ComponentOrFactory, Coordinate, Layer

DEG2RAD = np.pi / 180


def line(
    p_start: Union[Port, Coordinate],
    p_end: Union[Port, Coordinate],
    width: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if isinstance(p_start, pp.Port):
        width = p_start.width
        p_start = p_start.midpoint

    if isinstance(p_end, pp.Port):
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


@pp.cell_with_validator
def extend_ports(
    component: ComponentOrFactory = mmi1x2,
    port_list: Optional[List[str]] = None,
    length: float = 5.0,
    extension_factory: Optional[ComponentOrFactory] = None,
    extension_port_name_input: Optional[str] = None,
    extension_port_name_output: Optional[str] = None,
) -> Component:
    """Returns a new component with extended ports inside a container.

    Args:
        component: component to extend ports
        port_list: specify an list of ports names, if None it extends all ports
        length: extension length
        extension_factory: straight library to extend ports
        extension_port_name_input:
        extension_port_name_output:
    """
    c = pp.Component()
    component = component() if callable(component) else component
    c << component

    port_list = port_list or list(component.ports.keys())

    if extension_factory is None:
        dummy_port = component.ports[port_list[0]]

        def _ext_factory(length, width):
            return pp.components.hline(
                length=length, width=width, layer=dummy_port.layer
            )

        extension_factory = _ext_factory

    dummy_ext = extension_factory(length=length, width=0.5)
    port_labels = list(dummy_ext.ports.keys())

    extension_port_name_input = extension_port_name_input or port_labels[0]
    extension_port_name_output = extension_port_name_output or port_labels[-1]

    for port_name in component.ports.keys():
        if port_name in port_list:
            port = component.ports.get(port_name)
            extension_component = (
                extension_factory(length=length, width=port.width)
                if callable(extension_factory)
                else extension_factory
            )

            extension = c << extension_component
            extension.connect(extension_port_name_input, port)
            c.add_port(port_name, port=extension.ports[extension_port_name_output])
            c.absorb(extension)
        else:
            c.add_port(port_name, port=component.ports[port_name])
    return c


def test_extend_ports() -> Component:
    import pp.components as pc

    c = pc.straight(width=2)
    c = pc.cross(width=2)
    ce = extend_ports(component=c)
    assert len(c.ports) == len(ce.ports)
    return ce


def test_extend_ports_selection() -> Component:
    import pp.components as pc

    c = pc.cross(width=2)
    ce = extend_ports(component=c, port_list=["W0", "S0", "N0"])
    assert len(c.ports) == len(ce.ports)
    return ce


if __name__ == "__main__":
    c = extend_ports()
    # c = test_extend_ports_selection()
    c.show()

    # import pp.components as pc
    # c = pc.bend_circular()
    # ce = extend_ports(c, port_list=['W0'])

    # c = pc.straight(layer=(3, 0))
    # ce = extend_ports(c)
    # print(ce)
    # print(len(ce.ports))
    # pp.show(ce)
