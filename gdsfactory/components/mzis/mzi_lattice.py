from __future__ import annotations

__all__ = ["mzi_lattice", "mzi_lattice_mmi"]

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component

from .._schematic import mzi_2x2_schematic


@gf.cell_with_module_name(schematic_function=mzi_2x2_schematic, tags=["mzis"])
def mzi_lattice(
    coupler_lengths: Sequence[float] = (10.0, 20.0),
    coupler_gaps: Sequence[float] = (0.2, 0.3),
    delta_lengths: Sequence[float] = (10.0,),
    mzi: str = "mzi_coupler",
    splitter: str = "coupler",
    **kwargs: Any,
) -> Component:
    r"""Mzi lattice filter.

    Args:
        coupler_lengths: list of length for each coupler.
        coupler_gaps: list of coupler gaps.
        delta_lengths: list of length differences.
        mzi: function for the mzi.
        splitter: splitter function.
        kwargs: additional settings.

    Keyword Args:
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        cross_section: for routing (sxtop/sxbot to combiner).

    ```text
               ______             ______
              |      |           |      |
              |      |           |      |
         cp1==|      |===cp2=====|      |=== .... ===cp_last===
              |      |           |      |
              |      |           |      |
             DL1     |          DL2     |
              |      |           |      |
              |______|           |      |
                                 |______|
    ```

    """
    return cf.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
        mzi=mzi,
        splitter=splitter,
        **kwargs,
    )


@gf.cell_with_module_name(schematic_function=mzi_2x2_schematic, tags=["mzis"])
def mzi_lattice_mmi(
    coupler_widths: tuple[float | None, float | None] = (None, None),
    coupler_widths_tapers: tuple[float, ...] = (
        1.0,
        1.0,
    ),
    coupler_lengths_tapers: tuple[float, ...] = (
        10.0,
        10.0,
    ),
    coupler_lengths_mmis: tuple[float, ...] = (
        5.5,
        5.5,
    ),
    coupler_widths_mmis: tuple[float, ...] = (
        2.5,
        2.5,
    ),
    coupler_gaps_mmis: tuple[float, ...] = (
        0.25,
        0.25,
    ),
    taper_functions_mmis: tuple[str, ...] = (
        "taper",
        "taper",
    ),
    straight_functions_mmis: tuple[str, ...] = ("straight", "straight"),
    cross_sections_mmis: tuple[str, ...] = ("strip", "strip"),
    delta_lengths: tuple[float, ...] = (10.0,),
    mzi: str = "mzi2x2_2x2",
    splitter: str = "mmi2x2",
    **kwargs: Any,
) -> Component:
    r"""Mzi lattice filter, with MMI couplers.

    Args:
        coupler_widths: (for each MMI coupler, list of) input and output straight width.
        coupler_widths_tapers: (for each MMI coupler, list of) interface between input straights and mmi region.
        coupler_lengths_tapers: (for each MMI coupler, list of) into the mmi region.
        coupler_lengths_mmis: (for each MMI coupler, list of) in x direction.
        coupler_widths_mmis: (for each MMI coupler, list of) in y direction.
        coupler_gaps_mmis: (for each MMI coupler, list of) (width_taper + gap between tapered wg)/2.
        taper_functions_mmis: (for each MMI coupler, list of) taper function.
        straight_functions_mmis: (for each MMI coupler, list of) straight function.
        cross_sections_mmis: (for each MMI coupler, list of) spec.
        delta_lengths: list of length differences.
        mzi: function for the mzi.
        splitter: splitter function.
        kwargs: additional settings.

    Keyword Args:
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        cross_section: for routing (sxtop/sxbot to combiner).

    ```text
               ______             ______
              |      |           |      |
              |      |           |      |
         cp1==|      |===cp2=====|      |=== .... ===cp_last===
              |      |           |      |
              |      |           |      |
             DL1     |          DL2     |
              |      |           |      |
              |______|           |      |
                                 |______|
    ```

    """
    return cf.mzi_lattice_mmi(
        coupler_widths=coupler_widths,
        coupler_widths_tapers=coupler_widths_tapers,
        coupler_lengths_tapers=coupler_lengths_tapers,
        coupler_lengths_mmis=coupler_lengths_mmis,
        coupler_widths_mmis=coupler_widths_mmis,
        coupler_gaps_mmis=coupler_gaps_mmis,
        taper_functions_mmis=taper_functions_mmis,
        straight_functions_mmis=straight_functions_mmis,
        cross_sections_mmis=cross_sections_mmis,
        delta_lengths=delta_lengths,
        mzi=mzi,
        splitter=splitter,
        **kwargs,
    )
