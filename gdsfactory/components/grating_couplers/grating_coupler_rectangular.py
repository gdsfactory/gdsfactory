from __future__ import annotations

__all__ = ["grating_coupler_rectangular"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from .._schematic import grating_coupler_schematic


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_rectangular(
    n_periods: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = "taper",
    layer_slab: LayerSpec | None = "SLAB150",
    layer_grating: LayerSpec | None = None,
    fiber_angle: float = 15,
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Grating coupler with rectangular shapes (not elliptical).

    Needs longer taper than elliptical.
    Grating teeth are straight.
    For a focusing grating take a look at grating_coupler_elliptical.

    Args:
        n_periods: number of grating teeth.
        period: grating pitch.
        fill_factor: ratio of grating width vs gap.
        width_grating: 11.
        length_taper: 150.
        polarization: 'te' or 'tm'.
        wavelength: in um.
        taper: function.
        layer_slab: layer that protects the slab under the grating.
        layer_grating: layer for the grating.
        fiber_angle: in degrees.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
        cross_section: for input waveguide port.

        side view
                      fiber

                   /  /  /  /
                  /  /  /  /

    ```text
                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|
    ```


    ```text
        top view     _________
                    /| | | | |
                   / | | | | |
                  /taper_angle
                 /_ _| | | | |
        wg_width |   | | | | |
                 \   | | | | |
                  \  | | | | |
                   \ | | | | |
                    \|_|_|_|_|
                 <-->
                taper_length
    ```
    """
    return cf.grating_coupler_rectangular(
        n_periods=n_periods,
        period=period,
        fill_factor=fill_factor,
        width_grating=width_grating,
        length_taper=length_taper,
        polarization=polarization,
        wavelength=wavelength,
        taper=taper,
        layer_slab=layer_slab,
        layer_grating=layer_grating,
        fiber_angle=fiber_angle,
        slab_xmin=slab_xmin,
        slab_offset=slab_offset,
        cross_section=cross_section,
    )
