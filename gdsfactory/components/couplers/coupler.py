from __future__ import annotations

__all__ = ["coupler", "coupler_straight", "coupler_symmetric"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Delta

from .._schematic import coupler_schematic


@gf.cell_with_module_name(tags=["couplers"])
def coupler_symmetric(
    bend: ComponentSpec = "bend_s",
    gap: float = 0.234,
    dy: Delta = 4.0,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend spec.
        gap: in um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: section.
        allow_min_radius_violation: if True does not check for min bend radius.

    ```text
                       dx
                    |-----|
                       ___ o3
                      /       |
             o2 _____/        |
                              |
             o1 _____         |  dy
                     \        |
                      \___    |
                           o4
    ```

    """
    return cf.coupler_symmetric(
        bend=bend,
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Coupler_straight with two parallel straights.

    Args:
        length: of straight.
        gap: between straights.
        cross_section: specification (CrossSection, string or dict).

    ```text
        o2──────▲─────────o3
                │gap
        o1──────▼─────────o4
    ```
    """
    return cf.coupler_straight(
        length=length,
        gap=gap,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: Delta = 4.0,
    dx: Delta = 10.0,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    bend: ComponentSpec = "bend_s",
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights in um.
        length: of coupling region in um.
        dy: port to port vertical spacing in um.
        dx: length of bend in x direction in um.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True does not check for min bend radius.
        bend: input and output sbend components.

    ```text
               dx                                 dx
            |------|                           |------|
         o2 ________                           ______o3
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         o1                                          o4
    ```

                        coupler_straight  coupler_symmetric
    """
    return cf.coupler(
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        bend=bend,
    )
