"""Straight Doped PIN waveguide."""

from __future__ import annotations

__all__ = ["straight_pin_slot", "straight_pn_slot"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import modulator_schematic


@gf.cell_with_module_name(schematic_function=modulator_schematic, tags=["waveguides"])
def straight_pin_slot(
    length: float = 500.0,
    cross_section: CrossSectionSpec = "pin",
    via_stack: ComponentSpec | None = "via_stack_m1_mtop",
    via_stack_width: float = 10.0,
    via_stack_slab: ComponentSpec | None = "via_stack_slab_m1_horizontal",
    via_stack_slab_top: ComponentSpec | None = None,
    via_stack_slab_bot: ComponentSpec | None = None,
    via_stack_slab_width: float | None = None,
    via_stack_spacing: float = 3.0,
    via_stack_slab_spacing: float = 2.0,
    taper: ComponentSpec | None = "taper_strip_to_ridge",
    width: float | None = None,
) -> Component:
    """Returns a PIN straight waveguide with slotted via.

    <https://doi.org/10.1364/OE.26.029983>

    500um length for PI phase shift
    <https://ieeexplore.ieee.org/document/8268112>

    to go beyond 2PI, you will need at least 1mm
    <https://ieeexplore.ieee.org/document/8853396/>

    Args:
        length: of the waveguide.
        cross_section: for the waveguide.
        via_stack: for via_stacking the metal.
        via_stack_width: in um.
        via_stack_slab: function for the component via_stacking the slab.
        via_stack_slab_top: Optional, defaults to via_stack_slab.
        via_stack_slab_bot: Optional, defaults to via_stack_slab.
        via_stack_slab_width: defaults to via_stack_width.
        via_stack_spacing: spacing between via_stacks.
        via_stack_slab_spacing: spacing between via_stacks slabs.
        taper: optional taper.
        width: width of the waveguide. If None, it will use the width of the cross_section.
    """
    return cf.straight_pin_slot(
        length=length,
        cross_section=cross_section,
        via_stack=via_stack,
        via_stack_width=via_stack_width,
        via_stack_slab=via_stack_slab,
        via_stack_slab_top=via_stack_slab_top,
        via_stack_slab_bot=via_stack_slab_bot,
        via_stack_slab_width=via_stack_slab_width,
        via_stack_spacing=via_stack_spacing,
        via_stack_slab_spacing=via_stack_slab_spacing,
        taper=taper,
        width=width,
    )


straight_pn_slot = CellAlias(straight_pin_slot, cross_section="pn")

if __name__ == "__main__":
    c = straight_pin_slot()
    c.show()
