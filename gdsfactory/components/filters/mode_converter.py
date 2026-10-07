from __future__ import annotations

__all__ = ["bend_s_mode_converter", "mode_converter"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ckt_schematic
from ..bends.bend_s import bend_s

bend_s_mode_converter = CellAlias(bend_s, size=(25, 3))


@gf.cell_with_module_name(schematic_function=ckt_schematic, tags=["filters"])
def mode_converter(
    gap: float = 0.3,
    length: float = 10,
    coupler_straight_asymmetric: ComponentSpec = "coupler_straight_asymmetric",
    bend: ComponentSpec = "bend_s_mode_converter",
    taper: ComponentSpec = "taper",
    mm_width: float = 1.2,
    mc_mm_width: float = 1,
    sm_width: float = 0.5,
    taper_length: float = 25,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns Mode converter from TE0 to TE1.

    By matching the effective indices of two waveguides with different widths,
    light can couple from different transverse modes e.g. TE0 <-> TE1.
    <https://doi.org/10.1109/JPHOT.2019.2941742>

    Args:
        gap: directional coupler gap.
        length: coupler length interaction.
        coupler_straight_asymmetric: spec.
        bend: spec.
        taper: spec.
        mm_width: input/output multimode waveguide width.
        mc_mm_width: mode converter multimode waveguide width
        sm_width: single mode waveguide width.
        taper_length: taper length.
        cross_section: cross_section spec.

    ```text
        o2 ---           --- o4
              \         /
               \       /
                -------
        o1 -----=======----- o3
                |-----|
                length
    ```

        = : multimode width
        - : singlemode width
    """
    return cf.mode_converter(
        gap=gap,
        length=length,
        coupler_straight_asymmetric=coupler_straight_asymmetric,
        bend=bend,
        taper=taper,
        mm_width=mm_width,
        mc_mm_width=mc_mm_width,
        sm_width=sm_width,
        taper_length=taper_length,
        cross_section=cross_section,
    )
