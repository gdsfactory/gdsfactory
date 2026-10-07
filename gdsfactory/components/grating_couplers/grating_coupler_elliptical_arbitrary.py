from __future__ import annotations

__all__ = [
    "grating_coupler_elliptical_arbitrary",
    "grating_coupler_elliptical_uniform",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Floats, LayerSpec

from .._schematic import grating_coupler_schematic

_gaps = (0.1,) * 10
_widths = (0.5,) * 10


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_elliptical_arbitrary(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    taper_length: float = 16.6,
    taper_angle: float = 60.0,
    wavelength: float = 1.554,
    fiber_angle: float = 15.0,
    nclad: float = 1.443,
    layer_slab: LayerSpec | None = "SLAB150",
    layer_grating: LayerSpec | None = None,
    taper_to_slab_offset: float = -3.0,
    polarization: str = "te",
    spiked: bool = True,
    bias_gap: float = 0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    The ellipticity is derived from Lumerical knowledge base
    it depends on fiber_angle (degrees), neff, and nclad

    Args:
        gaps: list of gaps.
        widths: list of widths.
        taper_length: taper length from input.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        nclad: cladding effective index to compute ellipticity.
        layer_slab: Optional slab.
        layer_grating: Optional layer for grating.
            by default None uses cross_section.layer.
            if different from cross_section.layer expands taper.
        taper_to_slab_offset: 0 is where taper ends.
        polarization: te or tm.
        spiked: grating teeth have spikes to avoid drc errors.
        bias_gap: etch gap (um).
            Positive bias increases gap and reduces width to keep period constant.
        cross_section: cross_section spec for waveguide port.

    Notes:
        Ellipse conventions, see <https://en.wikipedia.org/wiki/Ellipse>

        c = (a1 ** 2 - b1 ** 2) ** 0.5
        e = (1 - (b1 / a1) ** 2) ** 0.5

    ```text
                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|
    ```
    """
    return cf.grating_coupler_elliptical_arbitrary(
        gaps=gaps,
        widths=widths,
        taper_length=taper_length,
        taper_angle=taper_angle,
        wavelength=wavelength,
        fiber_angle=fiber_angle,
        nclad=nclad,
        layer_slab=layer_slab,
        layer_grating=layer_grating,
        taper_to_slab_offset=taper_to_slab_offset,
        polarization=polarization,
        spiked=spiked,
        bias_gap=bias_gap,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_elliptical_uniform(
    n_periods: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    **kwargs: Any,
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    The ellipticity is derived from Lumerical knowledge base
    it depends on fiber_angle (degrees), neff, and nclad

    Args:
        n_periods: number of grating periods.
        period: grating pitch in um.
        fill_factor: ratio of grating width vs gap.

    Keyword Args:
        taper_length: taper length from input.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        neff: tooth effective index to compute ellipticity.
        nclad: cladding effective index to compute ellipticity.
        layer_slab: Optional slab.
        taper_to_slab_offset: where 0 is at the start of the taper.
        polarization: te or tm.
        spiked: grating teeth have spikes to avoid drc errors..
        bias_gap: etch gap (um).
            Positive bias increases gap and reduces width to keep period constant.
        cross_section: cross_section spec for waveguide port.
        kwargs: cross_section settings.

                      fiber

                   /  /  /  /
                  /  /  /  /

    ```text
                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|
    ```

    """
    return cf.grating_coupler_elliptical_uniform(
        n_periods=n_periods,
        period=period,
        fill_factor=fill_factor,
        **kwargs,
    )
