import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.taper import taper
from gdsfactory.components.text import text_rectangular
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Float2, Optional

edge_coupler_silicon = gf.partial(taper, width2=0.2, length=100, with_two_ports=False)


@gf.cell
def edge_coupler_array(
    edge_coupler: ComponentSpec = edge_coupler_silicon,
    n: int = 5,
    pitch: float = 127.0,
    h_mirror: bool = False,
    v_mirror: bool = False,
    text: Optional[ComponentSpec] = text_rectangular,
    text_offset: Float2 = (10, 20),
) -> Component:
    """Fiber array edge coupler based on an inverse taper.

    Each edge coupler adds a ruler for polishing.

    Args:
        edge_coupler: edge coupler spec.
        n: number of channels.
        pitch: Fiber pitch.
        h_mirror: horizontal mirror.
        v_mirror: vertical mirror.
        text: text spec.
        text_offset: from edge coupler.
    """
    edge_coupler = gf.get_component(edge_coupler)

    c = Component()
    for i in range(n):
        ref = edge_coupler.ref(h_mirror=h_mirror, v_mirror=v_mirror)
        c.aliases[i] = ref
        ref.y = i * pitch
        c.add(ref)
        for port in ref.get_ports_list():
            c.add_port(f"{port.name}_{i}", port=port)

        if text:
            t = c << gf.get_component(text, text=str(i + 1))
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
    h_mirror: bool = False,
    v_mirror: bool = False,
    right_loopback: bool = True,
    text: Optional[ComponentSpec] = text_rectangular,
    text_offset: Float2 = (0, 0),
) -> Component:
    """Fiber array edge coupler.

    Args:
        edge_coupler: edge coupler.
        cross_section: spec.
        radius: bend radius loopback (um).
        n: number of channels.
        pitch: Fiber pitch (um).
        extension_length: in um.
        h_mirror: horizontal mirror.
        v_mirror: vertical mirror.
        right_loopback: adds right loopback.
        text: Optional text spec.
        text_offset: x, y.
    """
    c = Component()
    ec = edge_coupler_array(
        edge_coupler=edge_coupler,
        n=n,
        pitch=pitch,
        h_mirror=h_mirror,
        v_mirror=v_mirror,
        text=text,
        text_offset=text_offset,
    )
    if extension_length > 0:
        ec = extend_ports(
            component=ec,
            port_names=("o1", "o2"),
            length=extension_length,
            extension=gf.partial(
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
    # c = edge_coupler_silicon()
    # c = edge_coupler_array()
    c = edge_coupler_array_with_loopback()
    c.show(show_ports=True)
