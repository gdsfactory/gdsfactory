from __future__ import annotations

__all__ = ["taper_adiabatic"]

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component_functions.tapers.taper_adiabatic import (
    neff_TE1550SOI_220nm,
)
from gdsfactory.typings import CrossSectionSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper_adiabatic(
    width1: float = 0.5,
    width2: float = 5.0,
    length: float = 0,
    neff_w: Callable[[float], float] = neff_TE1550SOI_220nm,
    alpha: float = 1,
    wavelength: float = 1.55,
    npoints: int = 200,
    cross_section: CrossSectionSpec = "strip",
    max_length: float = 200,
) -> gf.Component:
    """Returns a straight adiabatic_taper from an effective index callable.

    Args:
        width1: initial width.
        width2: final width.
        length: 0 uses the optimized length, and otherwise the optimal shape is compressed/stretched to the specified length.
        neff_w: a callable that returns the effective index as a function of width
                - By default, will use a compact model of neff(y) for fundamental 1550 nm TE mode of 220nm-thick core with 3.45 index, fully clad with 1.44 index. Many coefficients are needed to capture the behaviour.
        alpha: parameter that scales the rate of width change.
                - closer to 0 means longer and more adiabatic;
                - 1 is the intuitive limit beyond which higher order modes are excited;
                - [2] reports good performance up to 1.4 for fundamental TE in SOI (for multiple core thicknesses)
        wavelength: wavelength in um.
        npoints: number of points for sampling.
        cross_section: cross_section specification.
        max_length: maximum length for the taper.

    References:
        [1] Burns, W. K., et al. "Optical waveguide parabolic coupling horns." Appl. Phys. Lett., vol. 30, no. 1, 1 Jan. 1977, pp. 28-30, doi:10.1063/1.89199.
        [2] Fu, Yunfei, et al. "Efficient adiabatic silicon-on-insulator waveguide taper." Photonics Res., vol. 2, no. 3, 1 June 2014, pp. A41-A44, doi:10.1364/PRJ.2.000A41.
        npoints: number of points for sampling
    """
    return cf.taper_adiabatic(
        width1=width1,
        width2=width2,
        length=length,
        neff_w=neff_w,
        alpha=alpha,
        wavelength=wavelength,
        npoints=npoints,
        cross_section=cross_section,
        max_length=max_length,
    )
