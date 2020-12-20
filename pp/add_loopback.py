from pp.components import bend_circular, waveguide
from pp.routing.manhattan import round_corners


def gen_loopback(
    start_port,
    end_port,
    gc,
    grating_separation=127.0,
    gc_rotation=-90,
    gc_port_name="W0",
    bend_radius_align_ports=10.0,
    bend_factory=bend_circular,
    waveguide_factory=waveguide,
    y_bot_align_route=None,
):
    """
    Add a loopback (grating coupler align reference) to a start port and and end port
    Input grating generated on the left of start_port
    Output grating generated on the right of end_port

    .. code::

        __________________________________________
        | separation  |            |              |
        |             |            |              |
       GC          start_port  end_port          GC

    """

    gc = gc() if callable(gc) else gc

    if hasattr(start_port, "y"):
        y0 = start_port.y
    else:
        y0 = start_port[1]

    if hasattr(start_port, "x"):
        x0 = start_port.x - grating_separation
    else:
        x0 = start_port[0] - grating_separation

    if hasattr(end_port, "x"):
        x1 = end_port.x + grating_separation
    else:
        x1 = end_port[0] + grating_separation

    gca1, gca2 = [
        gc.ref(position=(x, y0), rotation=gc_rotation, port_id=gc_port_name)
        for x in [x0, x1]
    ]

    gsi = gc.size_info
    p0 = gca1.ports[gc_port_name].position
    p1 = gca2.ports[gc_port_name].position
    a = bend_radius_align_ports + 0.5
    b = max(2 * a, grating_separation / 2)
    y_bot_align_route = (
        y_bot_align_route if y_bot_align_route is not None else -gsi.width - 5.0
    )

    route = [
        p0,
        p0 + (0, a),
        p0 + (b, a),
        p0 + (b, y_bot_align_route),
        p1 + (-b, y_bot_align_route),
        p1 + (-b, a),
        p1 + (0, a),
        p1,
    ]
    bend90 = bend_factory(radius=bend_radius_align_ports)
    loop_back = round_corners(route, bend90, waveguide_factory)
    elements = [gca1, gca2, loop_back]
    return elements


if __name__ == "__main__":
    import pp

    c = waveguide()
    c.add(gen_loopback(c.ports["W0"], c.ports["E0"], gc=pp.c.grating_coupler_te))
    pp.show(c)
