import numpy as np

import pp

DEG2RAD = np.pi / 180


def line(p_start, p_end, width=None):
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


def move_polar_rad_copy(pos, angle, length):
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


def extend_ports(
    component,
    port_list=None,
    length=5,
    extension_factory=None,
    input_port_ext=None,
    output_port_ext=None,
    in_place=False,
):
    """ returns a component with extended ports """
    if in_place:
        c = component
    else:
        c = pp.Component(name=component.name + "_e")
        c << component

    if port_list is None:
        port_list = list(component.ports.keys())

    if extension_factory is None:
        dummy_port = component.ports[port_list[0]]

        def _ext_factory(length, width):
            return pp.c.hline(length=length, width=width, layer=dummy_port.layer)

        extension_factory = _ext_factory

    dummy_ext = extension_factory(length=length, width=0.5)
    port_labels = list(dummy_ext.ports.keys())
    if input_port_ext is None:
        input_port_ext = port_labels[0]

    if output_port_ext is None:
        output_port_ext = port_labels[-1]

    for port_label in port_list:
        port = component.ports.pop(port_label)

        extension = c << extension_factory(length=length, width=port.width)
        extension.connect(input_port_ext, port)
        c.add_port(port_label, port=extension.ports[output_port_ext])
    return c


if __name__ == "__main__":
    import pp.components as pc

    c = pc.bend_circular()
    ce = extend_ports(c)
    pp.show(ce)
