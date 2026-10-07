from __future__ import annotations

__all__ = [
    "cross_section_pn",
    "cross_section_rib",
    "ring_double_pn",
    "ring_single_pn",
    "via_stack_heater_ring_pn",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component_functions import CellAlias
from gdsfactory.component_functions.rings.ring_pn import (
    cross_section_pn,
    cross_section_rib,
)
from gdsfactory.cross_section import rib
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionFactory,
    CrossSectionSpec,
    LayerSpec,
)

from .._schematic import ring_double_schematic, ring_single_schematic
from ..vias.via import via
from ..vias.via_stack import via_stack

via_stack_heater_ring_pn = CellAlias(
    via_stack,
    size=(0.5, 0.5),
    layers=("M1", "M2", "M3"),
    vias=(
        CellAlias(via, layer="VIAC", size=(0.1, 0.1), enclosure=0.01, pitch=0.2),
        CellAlias(
            via,
            layer="VIA1",
            size=(0.1, 0.1),
            enclosure=0.01,
            pitch=0.2,
        ),
        None,
    ),
    correct_size=True,
)


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_double_pn(
    add_gap: float = 0.3,
    drop_gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 85,
    cross_section: CrossSectionFactory = rib,
    pn_cross_section: CrossSectionFactory = cross_section_pn,
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 2.175,
    heater_vias: ComponentSpec = "via_stack_heater_ring_pn",
    with_drop: bool = True,
    **kwargs: Any,
) -> gf.Component:
    """Returns add-drop pn ring with optional doped heater.

    Args:
        add_gap: gap to add waveguide. Bottom gap.
        drop_gap: gap to drop waveguide. Top gap.
        radius: for the bend and coupler.
        doping_angle: angle in degrees representing portion of ring that is doped.
        cross_section: cross_section spec for non-PN doped rib waveguide sections.
        pn_cross_section: cross section of pn junction.
        doped_heater: boolean for if we include doped heater or not.
        doped_heater_angle_buffer: angle in degrees buffering heater from pn junction.
        doped_heater_layer: doping layer for heater.
        doped_heater_width: width of doped heater.
        doped_heater_waveguide_offset: distance from the center of the ring waveguide to the center of the doped heater.
        heater_vias: components specifications for heater vias
        with_drop: boolean for if we include drop waveguide or not.
        kwargs: cross_section settings.

    """
    return cf.ring_double_pn(
        add_gap=add_gap,
        drop_gap=drop_gap,
        radius=radius,
        doping_angle=doping_angle,
        cross_section=cross_section,
        pn_cross_section=pn_cross_section,
        doped_heater=doped_heater,
        doped_heater_angle_buffer=doped_heater_angle_buffer,
        doped_heater_layer=doped_heater_layer,
        doped_heater_width=doped_heater_width,
        doped_heater_waveguide_offset=doped_heater_waveguide_offset,
        heater_vias=heater_vias,
        with_drop=with_drop,
        **kwargs,
    )


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def ring_single_pn(
    gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 250,
    cross_section: CrossSectionSpec = rib,
    pn_cross_section: CrossSectionSpec = cross_section_pn,
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 1.175,
    heater_vias: ComponentSpec = "via_stack_heater_ring_pn",
    pn_vias: ComponentSpec = "via_stack_slab_m3",
    pn_vias_width: float = 3,
    slab_simplify: float = 0.05,
) -> gf.Component:
    """Returns single pn ring with optional doped heater.

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        doping_angle: angle in degrees representing portion of ring that is doped.
        cross_section: cross_section spec for non-PN doped rib waveguide sections.
        pn_cross_section: cross section of pn junction.
        doped_heater: boolean for if we include doped heater or not.
        doped_heater_angle_buffer: angle in degrees buffering heater from pn junction.
        doped_heater_layer: doping layer for heater.
        doped_heater_width: width of doped heater.
        doped_heater_waveguide_offset: distance from the center of the ring waveguide to the center of the doped heater.
        heater_vias: components specifications for heater vias.
        pn_vias: components specifications for pn vias.
        pn_vias_width: width of pn vias.
        slab_simplify: polygon simplification tolerance for the undoped slab (um).
    """
    return cf.ring_single_pn(
        gap=gap,
        radius=radius,
        doping_angle=doping_angle,
        cross_section=cross_section,
        pn_cross_section=pn_cross_section,
        doped_heater=doped_heater,
        doped_heater_angle_buffer=doped_heater_angle_buffer,
        doped_heater_layer=doped_heater_layer,
        doped_heater_width=doped_heater_width,
        doped_heater_waveguide_offset=doped_heater_waveguide_offset,
        heater_vias=heater_vias,
        pn_vias=pn_vias,
        pn_vias_width=pn_vias_width,
        slab_simplify=slab_simplify,
    )


if __name__ == "__main__":
    c = ring_single_pn()
    c.show()
