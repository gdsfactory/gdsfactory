from __future__ import annotations

__all__ = [
    "grating_coupler_elliptical",
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import grating_coupler_schematic


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_elliptical(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 40.0,
    wavelength: float = 1.554,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,  # tooth effective index
    nclad: float = 1.443,
    n_periods: int = 30,
    big_last_tooth: bool = False,
    layer_slab: LayerSpec | None = "SLAB150",
    slab_xmin: float = -1.0,
    slab_offset: float = 2.0,
    spiked: bool = True,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    Args:
        polarization: te or tm.
        taper_length: taper length from input.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        grating_line_width: in um.
        neff: tooth effective index.
        nclad: cladding effective index.
        n_periods: number of periods.
        big_last_tooth: adds a big_last_tooth.
        layer_slab: layer that protects the slab under the grating.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: in um.
        spiked: grating teeth have sharp spikes to avoid non-manhattan drc errors.
        cross_section: specification (CrossSection, string or dict).

                      fiber

                   /  /  /  /
                  /  /  /  /

    ```text
                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|
    ```

    """
    return cf.grating_coupler_elliptical(
        polarization=polarization,
        taper_length=taper_length,
        taper_angle=taper_angle,
        wavelength=wavelength,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        neff=neff,
        nclad=nclad,
        n_periods=n_periods,
        big_last_tooth=big_last_tooth,
        layer_slab=layer_slab,
        slab_xmin=slab_xmin,
        slab_offset=slab_offset,
        spiked=spiked,
        cross_section=cross_section,
    )


grating_coupler_elliptical_tm = CellAlias(
    grating_coupler_elliptical,
    grating_line_width=0.707,
    polarization="tm",
    taper_length=30,
    slab_xmin=-2,
    neff=1.8,
    n_periods=16,
)


grating_coupler_elliptical_te = grating_coupler_elliptical
