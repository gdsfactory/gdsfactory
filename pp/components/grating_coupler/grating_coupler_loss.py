import inspect
import pp
from pp.components import grating_coupler_te
from pp.components import grating_coupler_tm
from pp.routing.connect import connect_strip
from pp.routing.manhattan import round_corners
from pp.add_labels import get_input_label


def connect_loop_back(port0, port1, a, b, R, y_bot_align_route):
    p0 = port0.position
    p1 = port1.position
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

    bend90 = pp.c.bend_circular(radius=R)
    loop_back = round_corners(route, bend90, pp.c.waveguide)
    return loop_back


@pp.autoname
def loss_deembedding_ch13_24(
    io_sep=127.0,
    R=10.0,
    grating_coupler_function=grating_coupler_te,
    input_port_indexes=[0, 1],
):

    gc = grating_coupler_function()
    c = pp.Component()
    dx = io_sep
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        connect_strip(gc_ports[0], gc_ports[2], start_straight=40.0, taper_factory=None)
    )

    gsi = gc.size_info
    p1 = gc_ports[1]
    p3 = gc_ports[3]
    a = R + 5.0  # 0.5
    b = max(2 * a, io_sep / 2)
    y_bot_align_route = -gsi.width - 5.0

    c.add(connect_loop_back(p1, p3, a, b, R, y_bot_align_route))
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)

    return c


@pp.autoname
def loss_deembedding_ch12_34(
    io_sep=127.0,
    R=10.0,
    grating_coupler_function=grating_coupler_te,
    input_port_indexes=[0, 2],
):
    gc = grating_coupler_function()

    c = pp.Component()
    dx = io_sep
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        connect_strip(gc_ports[0], gc_ports[1], start_straight=40.0, taper_factory=None)
    )
    c.add(
        connect_strip(gc_ports[2], gc_ports[3], start_straight=40.0, taper_factory=None)
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)
    return c


@pp.autoname
def loss_deembedding_ch14_23(
    io_sep=127.0,
    R=10.0,
    grating_coupler_function=grating_coupler_te,
    input_port_indexes=[0, 1],
):
    gc = grating_coupler_function()

    c = pp.Component()
    dx = io_sep
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        connect_strip(gc_ports[0], gc_ports[3], start_straight=40.0, taper_factory=None)
    )
    c.add(
        connect_strip(gc_ports[1], gc_ports[2], start_straight=30.0, taper_factory=None)
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)
    return c


@pp.autoname
def grating_coupler_loss_te(io_sep=127.0, grating_coupler_function=grating_coupler_te):
    c = pp.Component()
    _c1 = loss_deembedding_ch13_24(grating_coupler_function=grating_coupler_function)
    _c2 = loss_deembedding_ch14_23(grating_coupler_function=grating_coupler_function)
    _c3 = loss_deembedding_ch12_34(grating_coupler_function=grating_coupler_function)
    c.add_ref(_c1)
    c2 = c.add_ref(_c2)
    c3 = c.add_ref(_c3)
    c2.movex(io_sep * 4)
    c3.movex(io_sep * 8)

    return c


@pp.autoname
def grating_coupler_loss_tm(io_sep=127.0, grating_coupler_function=grating_coupler_tm):
    c = pp.Component()
    _c1 = loss_deembedding_ch13_24(grating_coupler_function=grating_coupler_function)
    _c2 = loss_deembedding_ch14_23(grating_coupler_function=grating_coupler_function)
    _c3 = loss_deembedding_ch12_34(grating_coupler_function=grating_coupler_function)
    c.add_ref(_c1)
    c2 = c.add_ref(_c2)
    c3 = c.add_ref(_c3)
    c2.movex(io_sep * 4)
    c3.movex(io_sep * 8)

    return c


if __name__ == "__main__":
    # c = loss_deembedding_ch14_23()
    # c = loss_deembedding_ch12_34()
    # c = loss_deembedding_ch13_24()
    c = grating_coupler_loss_te()
    # c = grating_coupler_loss_tm()
    pp.show(c)
