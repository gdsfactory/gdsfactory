from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.taper import taper
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

edge_coupler_silicon = partial(taper, width2=0.2, length=100, with_two_ports=True)


@gf.cell
def edge_coupler_array(
    edge_coupler: ComponentSpec = "edge_coupler_silicon",
    n: int = 5,
    pitch: float = 127.0,
    x_reflection: bool = False,
    text: ComponentSpec | None = "text_rectangular",
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
        ref = c.add_ref(edge_coupler)
        ref.name = f"ec_{i}"
        ref.dy = i * pitch

        if x_reflection:
            ref.dmirror()

        # for port in ref.ports:
        #     c.add_port(f"{port.name}_{i}", port=port)

        c.add_ports(ref.ports, prefix=str(i))

        if text:
            t = c << gf.get_component(text, text=str(i + 1))
            t.drotate(text_rotation)
            t.dmovex(text_offset[0])
            t.dmovey(i * pitch + text_offset[1])

    c.auto_rename_ports()
    return c


@gf.cell
def edge_coupler_array_with_loopback(
    edge_coupler: ComponentSpec = "edge_coupler_silicon",
    cross_section: CrossSectionSpec = "strip",
    radius: float = 30,
    n: int = 8,
    pitch: float = 127.0,
    extension_length: float = 1.0,
    x_reflection: bool = False,
    text: ComponentSpec | None = "text_rectangular",
    text_offset: Float2 = (0, 10),
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
    p1 = ec_ref.ports["o1"]
    p2 = ec_ref.ports["o2"]
    p3 = ec_ref.ports[f"o{n-1}"]
    p4 = ec_ref.ports[f"o{n}"]

    gf.routing.route_single(
        c,
        p1,
        p2,
        cross_section=cross_section,
        radius=radius,
    )
    gf.routing.route_single(
        c,
        p3,
        p4,
        cross_section=cross_section,
        radius=radius,
    )

    for i, port in enumerate(ec_ref.ports):
        if port not in [p1, p2, p3, p4]:
            c.add_port(str(i), port=port)

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c = edge_coupler_array()
    c = edge_coupler_array_with_loopback()
    # c = edge_coupler_array_with_loopback(text_rotation=90)
    # c = edge_coupler_silicon()
    # c = edge_coupler_array(x_reflection=False)
    # c = edge_coupler_array_with_loopback(x_reflection=False)
    c.show()
