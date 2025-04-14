from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2


@gf.cell_with_module_name
def edge_coupler_silicon(
    length: float = 100,
    width1: float = 0.5,
    width2: float = 0.2,
    with_two_ports: bool = True,
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "edge_coupler"),
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Edge coupler for silicon photonics.

    Args:
        length: length of the taper.
        width1: width1 of the taper.
        width2: width2 of the taper.
        with_two_ports: add two ports.
        port_names: tuple with port names.
        port_types: tuple with port types.
        cross_section: cross_section spec.

    """
    return gf.c.taper(
        width1=width1,
        width2=width2,
        length=length,
        with_two_ports=with_two_ports,
        port_names=port_names,
        port_types=port_types,
        cross_section=cross_section,
    )


@gf.cell_with_module_name
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
        ref.y = i * pitch

        if x_reflection:
            ref.dmirror()

        # for port in ref.ports:
        #     c.add_port(f"{port.name}_{i}", port=port)

        c.add_ports(ref.ports, prefix=str(i))

        if text:
            t = c << gf.get_component(text, text=str(i + 1))
            t.rotate(text_rotation)
            t.movex(text_offset[0])
            t.movey(i * pitch + text_offset[1])

    c.auto_rename_ports()
    return c


@gf.cell_with_module_name
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
        ec = gf.c.extend_ports(
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
    p3 = ec_ref.ports[f"o{n - 1}"]
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
    c = edge_coupler_silicon()
    # c = edge_coupler_array_with_loopback(text_rotation=90)
    # c = edge_coupler_silicon()
    # c = edge_coupler_array(x_reflection=False)
    # c = edge_coupler_array_with_loopback(x_reflection=False)
    c.show()
