from __future__ import annotations

import inspect
from typing import Callable, List, Tuple

import gdsfactory as gf
from gdsfactory.add_labels import get_input_label as get_input_label_function
from gdsfactory.cell import cell
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.port import Port
from gdsfactory.routing.get_route import get_route
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


def connect_loopback(
    port0: Port,
    port1: Port,
    a: float,
    b: float,
    y_bot_align_route: float,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> List[ComponentReference]:
    """Connects loopback structure."""
    p0 = port0.center
    p1 = port1.center
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

    bend90 = gf.components.bend_euler(cross_section=cross_section, **kwargs)
    return round_corners(
        points=points,
        bend=bend90,
        straight=gf.components.straight,
        cross_section=cross_section,
        **kwargs,
    ).references


@cell
def loss_deembedding_ch13_24(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    input_port_indexes: Tuple[int, ...] = (0, 1),
    cross_section: CrossSectionSpec = "strip",
    port_name: str = "o1",
    get_input_label: Callable = get_input_label_function,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->3, 2->4

    Args:
        pitch: um.
        grating_coupler: spec.
        input_port_indexes: adds test labels.
        cross_section: spec.
        port_name: for the grating_coupler port.
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)
    c = gf.Component()
    dx = pitch
    gcs = [
        gc.ref(position=(i * dx, 0), port_id=port_name, rotation=-90) for i in range(4)
    ]

    gc_ports = [g.ports[port_name] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[2],
            start_straight_length=40.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )

    x = gf.get_cross_section(cross_section, **kwargs)
    radius = x.radius

    gsi = gc.size_info
    p1 = gc_ports[1]
    p3 = gc_ports[3]
    a = radius + 5.0  # 0.5
    b = max(2 * a, pitch / 2)
    y_bot_align_route = -gsi.width - 5.0

    c.add(
        connect_loopback(
            p1, p3, a, b, y_bot_align_route, cross_section=cross_section, **kwargs
        )
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.origin = gc_ports[index].center
        c.add(label)

    return c


@cell
def loss_deembedding_ch12_34(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    input_port_indexes: Tuple[int, ...] = (0, 2),
    port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    get_input_label: Callable = get_input_label_function,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->2, 3->4

    Args:
        pitch: um.
        grating_coupler: spec.
        input_port_indexes: for grating couplers.
        port_name: for the grating_coupler port.
        cross_section: spec.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch
    gcs = [
        gc.ref(position=(i * dx, 0), port_id=port_name, rotation=-90) for i in range(4)
    ]

    gc_ports = [g.ports[port_name] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[1],
            start_straight_length=40.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )
    c.add(
        get_route(
            gc_ports[2],
            gc_ports[3],
            start_straight_length=40.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.origin = gc_ports[index].center
        c.add(label)
    return c


@cell
def loss_deembedding_ch14_23(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    input_port_indexes: Tuple[int, ...] = (0, 1),
    cross_section: CrossSectionSpec = "strip",
    port_name: str = "o1",
    get_input_label: Callable = get_input_label_function,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->4, 2->3

    Args:
        pitch: um.
        grating_coupler: spec.
        input_port_indexes: for grating couplers.
        cross_section: spec.
        port_name: for the grating_coupler port.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch
    gcs = [
        gc.ref(position=(i * dx, 0), port_id=port_name, rotation=-90) for i in range(4)
    ]

    gc_ports = [g.ports[port_name] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[3],
            start_straight_length=40.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )
    c.add(
        get_route(
            gc_ports[1],
            gc_ports[2],
            start_straight_length=30.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.origin = gc_ports[index].center
        c.add(label)
    return c


@cell
def grating_coupler_loss_fiber_array(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = grating_coupler_te,
    input_port_indexes: Tuple[int, ...] = (0, 1),
    port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    get_input_label: Callable = get_input_label_function,
    **kwargs,
) -> Component:
    """Returns Grating coupler fiber array loopback.

    Args:
        pitch: spacing.
        grating_coupler: spec for grating coupler.
        input_port_indexes: for grating couplers.
        port_name: for the grating_coupler port.
        cross_section: spec.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch
    gcs = [
        gc.ref(position=(i * dx, 0), port_id=port_name, rotation=-90) for i in range(2)
    ]

    gc_ports = [g.ports[port_name] for g in gcs]
    c.add(gcs)

    c.add(
        get_route(
            gc_ports[0],
            gc_ports[1],
            start_straight_length=40.0,
            taper=None,
            cross_section=cross_section,
            **kwargs,
        ).references
    )
    for i, index in enumerate(input_port_indexes):
        label = get_input_label(
            gc_ports[index], gc, i, component_name=inspect.stack()[0][3]
        )
        label.origin = gc_ports[index].center
        c.add(label)
    return c


@cell
def grating_coupler_loss_fiber_array4(
    pitch: float = 127.0, grating_coupler: ComponentSpec = grating_coupler_te, **kwargs
) -> Component:
    """Returns a grating coupler test structure for fiber array.

    Measures all combinations for a 4 fiber fiber_array

    Connects channel 1->3, 2->4
    Connects channel 1->4, 2->3
    Connects channel 1->2, 3->4

    Args:
        pitch: grating_coupler_pitch.
        grating_coupler: function.
        kwargs: cross_section settings.
    """
    c = gf.Component()
    c1 = loss_deembedding_ch13_24(grating_coupler=grating_coupler, **kwargs)
    c2 = loss_deembedding_ch14_23(grating_coupler=grating_coupler, **kwargs)
    c3 = loss_deembedding_ch12_34(grating_coupler=grating_coupler, **kwargs)
    c.add_ref(c1)
    c2 = c.add_ref(c2)
    c3 = c.add_ref(c3)
    c2.movex(pitch * 4)
    c3.movex(pitch * 8)
    return c


if __name__ == "__main__":
    # c = loss_deembedding_ch14_23()
    # c = loss_deembedding_ch12_34()
    # c = loss_deembedding_ch13_24()
    # c = grating_coupler_loss_fiber_array4(layer=(2, 0), radius=30)
    c = grating_coupler_loss_fiber_array4(cross_section="rib")
    # c = grating_coupler_loss_fiber_array(layer=(2, 0), radius=30)
    c.show(show_ports=True)
