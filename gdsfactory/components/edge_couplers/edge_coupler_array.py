from __future__ import annotations

__all__ = [
    "edge_coupler_array",
    "edge_coupler_array_with_loopback",
    "edge_coupler_silicon",
]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["edge_couplers"])
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
    return cf.edge_coupler_silicon(
        length=length,
        width1=width1,
        width2=width2,
        with_two_ports=with_two_ports,
        port_names=port_names,
        port_types=port_types,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(tags=["edge_couplers"])
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
    return cf.edge_coupler_array(
        edge_coupler=edge_coupler,
        n=n,
        pitch=pitch,
        x_reflection=x_reflection,
        text=text,
        text_offset=text_offset,
        text_rotation=text_rotation,
    )


@gf.cell_with_module_name(tags=["edge_couplers"])
def edge_coupler_array_with_loopback(
    edge_coupler: ComponentSpec = "edge_coupler_silicon",
    cross_section: CrossSectionSpec = "strip",
    radius: float | None = None,
    n: int = 8,
    pitch: float = 127.0,
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
        x_reflection: horizontal mirror.
        text: Optional text spec.
        text_offset: x, y.
        text_rotation: text rotation in degrees.
    """
    return cf.edge_coupler_array_with_loopback(
        edge_coupler=edge_coupler,
        cross_section=cross_section,
        radius=radius,
        n=n,
        pitch=pitch,
        x_reflection=x_reflection,
        text=text,
        text_offset=text_offset,
        text_rotation=text_rotation,
    )


if __name__ == "__main__":
    # c = edge_coupler_array(x_reflection=True, port_orientation=0)
    # c = edge_coupler_array(x_reflection=False, port_orientation=180)
    c = edge_coupler_array_with_loopback(
        n=5,
        pitch=127.0,
        x_reflection=False,
        text="text_rectangular",
        text_offset=(0, 10),
        text_rotation=0,
    )
    c.pprint_ports()
    c.show()
