import numpy as np
from pp.container import container

import pp
from numpy import float64, ndarray
from typing import List, Optional

DEG2RAD = np.pi / 180


def line(
    p_start: ndarray, p_end: ndarray, width: Optional[float] = None
) -> List[ndarray]:
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
    return [p0, p1, p2, p3]


def move_polar_rad_copy(pos: ndarray, angle: float64, length: float) -> ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return pos + length * np.array([c, s])


@pp.autoname
def extend_port(port, length):
    """ returns a port extended by length """
    c = pp.Component()

    # Generate a port extension
    p_start = port.midpoint
    angle = port.angle
    p_end = move_polar_rad_copy(p_start, angle * DEG2RAD, length)
    w = port.width

    _line = line(p_start, p_end, w)

    c.add_polygon(_line, layer=port.layer)
    c.add_port(name="original", port=port)
    c.add_port(
        name=port.name,
        midpoint=p_end,
        width=port.width,
        orientation=port.orientation,
        layer=port.layer,
    )

    return c


@container
def extend_ports(
    component,
    port_list=None,
    length=5,
    extension_factory=None,
    input_port_ext=None,
    output_port_ext=None,
):
    """ returns a component with extended ports
    """
    c = pp.Component(name=component.name + "_e")
    c << component

    port_list = port_list or list(component.ports.keys())

    if extension_factory is None:
        dummy_port = component.ports[port_list[0]]

        def _ext_factory(length, width):
            return pp.c.hline(length=length, width=width, layer=dummy_port.layer)

        extension_factory = _ext_factory

    dummy_ext = extension_factory(length=length, width=0.5)
    port_labels = list(dummy_ext.ports.keys())

    input_port_ext = input_port_ext or port_labels[0]
    output_port_ext = output_port_ext or port_labels[-1]

    for port_label in port_list:
        port = component.ports.get(port_label)
        extension = c << extension_factory(length=length, width=port.width)
        extension.connect(input_port_ext, port)
        c.add_port(port_label, port=extension.ports[output_port_ext])
    return c


def test_extend_ports():
    import pp.components as pc

    c = pc.waveguide()
    ce = extend_ports(c)
    assert len(c.ports) == len(ce.ports)


if __name__ == "__main__":
    test_extend_ports()
    import pp.components as pc

    # c = pc.bend_circular()
    # ce = extend_ports(c, port_list=['W0'])

    c = pc.waveguide(layer=(3, 0))
    ce = extend_ports(c)
    print(ce)
    print(len(ce.ports))
    pp.show(ce)
