from __future__ import annotations

__all__ = ["mzit", "mzit_lattice"]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Delta

from .._schematic import mzi_2x2_schematic


@gf.cell_with_module_name(schematic_function=mzi_2x2_schematic, tags=["mzis"])
def mzit(
    w0: float = 0.5,
    w1: float = 0.45,
    w2: float = 0.55,
    dy: Delta = 2.0,
    delta_length: float = 10.0,
    length: float = 1.0,
    coupler_length1: float = 5.0,
    coupler_length2: float = 10.0,
    coupler_gap1: float = 0.2,
    coupler_gap2: float = 0.3,
    taper: ComponentSpec = "taper",
    taper_length: float = 5.0,
    bend90: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler1: ComponentSpec | None = "coupler",
    coupler2: ComponentSpec = "coupler",
    cross_section: str = "strip",
) -> Component:
    r"""Mzi tolerant to fabrication variations.

    based on Yufei Xing thesis
    <http://photonics.intec.ugent.be/publications/PhD.asp?ID=250>

    Args:
        w0: input waveguide width (um).
        w1: narrow waveguide width (um).
        w2: wide waveguide width (um).
        dy: port to port vertical spacing.
        delta_length: length difference between arms (um).
        length: shared length for w1 and w2.
        coupler_length1: length of coupler1.
        coupler_length2: length of coupler2.
        coupler_gap1: coupler1.
        coupler_gap2: coupler2.
        taper: taper spec.
        taper_length: from w0 to w1.
        bend90: bend spec.
        straight: spec.
        coupler1: coupler1 spec (optional).
        coupler2: coupler2 spec.
        cross_section: cross_section spec.

    ```text
                           cp1
            4   2 __                  __  3___w0_t2   _w2___
                    \                /                      \
                     \    length1   /                        |
                      ============== gap1                    |
                     /              \                        |
                  __/                \_____w0___t1   _w1     |
            3   1                        4               \   |
                                                         |   |
            2   2                                        |   |
                  __                  __w0____t1____w1___/   |
                    \                /                       |
                     \    length2   /                        |
                      ============== gap2                    |
                     /               \                       |                       |
                  __/                 \ E0_w0__t2 __w1______/
            1   1
                           cp2
    ```


    """
    return cf.mzit(
        w0=w0,
        w1=w1,
        w2=w2,
        dy=dy,
        delta_length=delta_length,
        length=length,
        coupler_length1=coupler_length1,
        coupler_length2=coupler_length2,
        coupler_gap1=coupler_gap1,
        coupler_gap2=coupler_gap2,
        taper=taper,
        taper_length=taper_length,
        bend90=bend90,
        straight=straight,
        coupler1=coupler1,
        coupler2=coupler2,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(schematic_function=mzi_2x2_schematic, tags=["mzis"])
def mzit_lattice(
    coupler_lengths: Sequence[float] = (10.0, 20.0),
    coupler_gaps: Sequence[float] = (0.2, 0.3),
    delta_lengths: Sequence[float] = (10.0,),
    mzi: ComponentSpec = "mzit",
) -> Component:
    r"""Mzi fab tolerant lattice filter.

    Args:
        coupler_lengths: list of coupler lengths, in um. One per coupler.
        coupler_gaps: list of coupler gaps, in um. One per coupler.
        delta_lengths: list of length differences between the MZI arms, in um.
            One per MZI, so one less than the number of couplers.
        mzi: MZI component spec.

    ```text
                    cp1
    o4  o2 __                  __ o3___w0_t2   _w2___
             \                /                      \
                     \    length1   /                        |
               ============== gap1                    |
              /              \                        |
           __/                \_____w0___t1   _w1     |
    o3  o1                       o4               \   | .
                     ...                          |   | .
    o2  o2                    o3                  |   | .
           __                  _____w0___t1___w1__/   |
             \                /                       |
              \    lengthN   /                        |
               ============== gapN                    |
              /               \                       |
           __/                 \_                     |
    o1  o1                      \___w0___t2___w1_____/
                    cpN       o4
    ```


    """
    return cf.mzit_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
        mzi=mzi,
    )
