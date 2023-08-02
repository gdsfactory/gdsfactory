from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.taper import taper
from gdsfactory.components.text import text_rectangular
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

edge_coupler_silicon = partial(taper, width2=0.2, length=100, with_two_ports=False)


@gf.cell
def edge_coupler_array(
    edge_coupler: ComponentSpec = edge_coupler_silicon,
    n: int = 5,
    pitch: float = 127.0,
    x_reflection: bool = False,
    text: ComponentSpec | None = text_rectangular,
    text_offset: Float2 = (10, 20),
    text_rotation: float = 0,
) -> Component:
    """Fiber array edge coupler based on an inverse taper.

    Each edge coupler adds a ruler for polishing.

    Args:
        edge_coupler: edge coupler spec.
        n: number of channels.
        pitch: Fiber pitch.
        x_reflection: horizontal mirror.
        text: text spec.
        text_offset: from edge coupler.
        text_rotation: text rotation in degrees.
    """
    edge_coupler = gf.get_component(edge_coupler)

    c = Component()
    for i in range(n):
        alias = f"ec_{i}"
        ref = c.add_ref(edge_coupler, alias=alias)
        ref.y = i * pitch

        if x_reflection:
            ref.mirror()

        for port in ref.get_ports_list():
            c.add_port(f"{port.name}_{i}", port=port)

        if text:
            t = c << gf.get_component(text, text=str(i + 1))
            t.rotate(text_rotation)
            t.move(np.array(text_offset) + (0, i * pitch))

    c.auto_rename_ports()
    return c


@gf.cell
def edge_coupler_array_with_loopback(
    edge_coupler: ComponentSpec = edge_coupler_silicon,
    cross_section: CrossSectionSpec = "strip",
    radius: float = 30,
    n: int = 8,
    pitch: float = 127.0,
    extension_length: float = 1.0,
    right_loopback: bool = True,
    x_reflection: bool = False,
    text: ComponentSpec | None = text_rectangular,
    text_offset: Float2 = (0, 0),
    text_rotation: float = 0,
) -> Component:
    """Fiber array edge coupler.

    Args:
        edge_coupler: edge coupler.
        cross_section: spec.
        radius: bend radius loopback (um).
        n: number of channels.
        pitch: Fiber pitch (um).
        extension_length: in um.
        right_loopback: adds right loopback.
        x_reflection: horizontal mirror.
        text: Optional text spec.
        text_offset: x, y.
        text_rotation: text rotation in degrees.
    """
    c = Component()
    ec = edge_coupler_array(
        edge_coupler=edge_coupler,
        n=n,
        pitch=pitch,
        x_reflection=x_reflection,
        text=text,
        text_offset=text_offset,
        text_rotation=text_rotation,
    )
    if extension_length > 0:
        ec = extend_ports(
            component=ec,
            port_names=("o1", "o2"),
            length=extension_length,
            extension=partial(
                gf.c.straight, cross_section=cross_section, length=extension_length
            ),
        )

    ec_ref = c << ec
    route1 = gf.routing.get_route(
        ec_ref.ports["o1"],
        ec_ref.ports["o2"],
        cross_section=cross_section,
        radius=radius,
    )
    c.add(route1.references)

    if n > 4 and right_loopback:
        route2 = gf.routing.get_route(
            ec_ref.ports[f"o{n-1}"],
            ec_ref.ports[f"o{n}"],
            cross_section=cross_section,
            radius=radius,
        )
        c.add(route2.references)
        for i in range(n - 4):
            c.add_port(str(i), port=ec_ref.ports[f"o{i+3}"])
    elif n > 4:
        for i in range(n - 2):
            c.add_port(str(i), port=ec_ref.ports[f"o{i+3}"])

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = edge_coupler_array_with_loopback(text_rotation=90)
    # c = edge_coupler_silicon()
    # c = edge_coupler_array(x_reflection=False)
    # c = edge_coupler_array_with_loopback(x_reflection=False)
    c.show(show_ports=True)
