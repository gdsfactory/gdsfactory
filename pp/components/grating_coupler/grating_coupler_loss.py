import inspect
from typing import Iterable

import pp
from pp.add_labels import get_input_label
from pp.cell import cell
from pp.component import Component, ComponentReference
from pp.components import grating_coupler_te
from pp.port import Port
from pp.routing.get_route import get_route
from pp.routing.manhattan import round_corners
from pp.types import ComponentFactory


def connect_loop_back(
    port0: Port,
    port1: Port,
    a: float,
    b: float,
    R: float,
    y_bot_align_route: float,
    waveguide: str = "strip",
    **kwargs
) -> ComponentReference:
    p0 = port0.position
    p1 = port1.position
    points = [
        p0,
        p0 + (0, a),
        p0 + (b, a),
        p0 + (b, y_bot_align_route),
        p1 + (-b, y_bot_align_route),
        p1 + (-b, a),
        p1 + (0, a),
        p1,
    ]

    bend90 = pp.components.bend_euler(radius=R, waveguide=waveguide, **kwargs)
    return round_corners(
        points=points,
        bend_factory=bend90,
        straight_factory=pp.components.straight,
        waveguide=waveguide,
        **kwargs
    ).references


@cell
def loss_deembedding_ch13_24(
    pitch: float = 127.0,
    R: float = 10.0,
    grating_coupler_factory: ComponentFactory = grating_coupler_te,
    input_port_indexes: Iterable[int] = (0, 1),
    waveguide: str = "strip",
    **kwargs
) -> Component:

    gc = grating_coupler_factory()
    c = pp.Component()
    dx = pitch
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[2],
            start_straight=40.0,
            taper_factory=None,
            waveguide=waveguide,
            **kwargs
        ).references
    )

    gsi = gc.size_info
    p1 = gc_ports[1]
    p3 = gc_ports[3]
    a = R + 5.0  # 0.5
    b = max(2 * a, pitch / 2)
    y_bot_align_route = -gsi.width - 5.0

    c.add(
        connect_loop_back(
            p1, p3, a, b, R, y_bot_align_route, waveguide=waveguide, **kwargs
        )
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)

    return c


@cell
def loss_deembedding_ch12_34(
    pitch: float = 127.0,
    R: float = 10.0,
    grating_coupler_factory: ComponentFactory = grating_coupler_te,
    input_port_indexes: Iterable[int] = (0, 2),
    waveguide: str = "strip",
    **kwargs
) -> Component:
    gc = grating_coupler_factory()

    c = pp.Component()
    dx = pitch
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[1],
            start_straight=40.0,
            taper_factory=None,
            waveguide=waveguide,
            **kwargs
        ).references
    )
    c.add(
        get_route(
            gc_ports[2],
            gc_ports[3],
            start_straight=40.0,
            taper_factory=None,
            waveguide=waveguide,
            **kwargs
        ).references
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)
    return c


@cell
def loss_deembedding_ch14_23(
    pitch: float = 127.0,
    R: float = 10.0,
    grating_coupler_factory: ComponentFactory = grating_coupler_te,
    input_port_indexes: Iterable[int] = (0, 1),
    waveguide: str = "strip",
    **kwargs
) -> Component:
    gc = grating_coupler_factory()

    c = pp.Component()
    dx = pitch
    gcs = [gc.ref(position=(i * dx, 0), port_id="W0", rotation=-90) for i in range(4)]

    gc_ports = [g.ports["W0"] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[3],
            start_straight=40.0,
            taper_factory=None,
            waveguide=waveguide,
            **kwargs
        ).references
    )
    c.add(
        get_route(
            gc_ports[1],
            gc_ports[2],
            start_straight=30.0,
            taper_factory=None,
            waveguide=waveguide,
            **kwargs
        ).references
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.position = gc_ports[index].position
        c.add(label)
    return c


@cell
def grating_coupler_loss(
    pitch: float = 127.0,
    grating_coupler_factory: ComponentFactory = grating_coupler_te,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    """

    Args:
        pitch: grating_coupler_pitch
        grating_coupler_factory: function
        waveguide: from TECH.waveguide
        **kwargs: waveguide_settings

    """
    c = pp.Component()
    _c1 = loss_deembedding_ch13_24(
        grating_coupler_factory=grating_coupler_factory, waveguide=waveguide, **kwargs
    )
    _c2 = loss_deembedding_ch14_23(
        grating_coupler_factory=grating_coupler_factory, waveguide=waveguide, **kwargs
    )
    _c3 = loss_deembedding_ch12_34(
        grating_coupler_factory=grating_coupler_factory, waveguide=waveguide, **kwargs
    )
    c.add_ref(_c1)
    c2 = c.add_ref(_c2)
    c3 = c.add_ref(_c3)
    c2.movex(pitch * 4)
    c3.movex(pitch * 8)

    return c


if __name__ == "__main__":
    # c = loss_deembedding_ch14_23()
    # c = loss_deembedding_ch12_34()
    # c = loss_deembedding_ch13_24()
    c = grating_coupler_loss(waveguide="nitride")
    c.show()
