"""Wires for electrical manhattan routes."""

from __future__ import annotations

__all__ = [
    "wire_corner",
    "wire_corner45",
    "wire_corner45_straight",
    "wire_corner_sections",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec, PortNames, PortTypes


@gf.cell_with_module_name(tags=["waveguides"])
def wire_corner(
    cross_section: CrossSectionSpec = "metal_routing",
    port_names: PortNames = ("e1", "e2"),
    port_types: PortTypes = ("electrical", "electrical"),
    width: float | None = None,
    radius: float | None = None,
) -> Component:
    """Returns 45 degrees electrical corner wire.

    Args:
        cross_section: spec.
        port_names: port names.
        port_types: port types.
        width: optional width. Defaults to cross_section width.
        radius: ignored.
    """
    return cf.wire_corner(
        cross_section=cross_section,
        port_names=port_names,
        port_types=port_types,
        width=width,
        radius=radius,
    )


@gf.cell(tags=["waveguides"])
def wire_corner45_straight(
    width: float | None = None,
    radius: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Returns 45 degrees wire straight ends.

    Args:
        width: of the wire.
        radius: of the corner. Defaults to width.
        cross_section: metal_routing.
    """
    return cf.wire_corner45_straight(
        width=width, radius=radius, cross_section=cross_section
    )


@gf.cell_with_module_name(tags=["waveguides"])
def wire_corner45(
    cross_section: CrossSectionSpec = "metal_routing",
    radius: float = 10,
    width: float | None = None,
    layer: LayerSpec | None = None,
    with_corner90_ports: bool = True,
) -> Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        cross_section: spec.
        radius: in um.
        width: optional width.
        layer: optional layer.
        with_corner90_ports: if True adds ports at 90 degrees.
    """
    return cf.wire_corner45(
        cross_section=cross_section,
        radius=radius,
        width=width,
        layer=layer,
        with_corner90_ports=with_corner90_ports,
    )


@gf.cell_with_module_name(tags=["waveguides"])
def wire_corner_sections(
    cross_section: CrossSectionSpec = "metal_routing",
    port_type: str = "electrical",
    **kwargs: Any,
) -> Component:
    """Returns 90 degrees electrical corner wire, where all cross_section sections properly represented.

    Works well with symmetric cross_sections, not quite ready for asymmetric.

    Args:
        cross_section: spec.
        port_type: "electrical" or "optical".
        kwargs: cross_section settings, ignored (such as radius, width, layer).
    """
    return cf.wire_corner_sections(
        cross_section=cross_section, port_type=port_type, **kwargs
    )
