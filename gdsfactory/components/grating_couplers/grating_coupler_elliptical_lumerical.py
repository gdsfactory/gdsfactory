from __future__ import annotations

__all__ = [
    "grating_coupler_elliptical_lumerical",
    "grating_coupler_elliptical_lumerical_etch70",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.component_functions.grating_couplers.grating_coupler_elliptical_lumerical import (
    parameters,
)
from gdsfactory.typings import CrossSectionSpec, Floats, LayerSpec

from .._schematic import grating_coupler_schematic


@gf.cell_with_module_name(
    schematic_function=grating_coupler_schematic, tags=["grating_couplers"]
)
def grating_coupler_elliptical_lumerical(
    parameters: Floats = parameters,
    layer_slab: LayerSpec | None = "SLAB150",
    taper_angle: float = 55,
    taper_length: float = 12.24 + 0.36,
    fiber_angle: float = 5,
    info: dict[str, Any] | None = None,
    bias_gap: float = 0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a grating coupler from lumerical inverse design 3D optimization.

    this is a wrapper of components.grating_coupler_elliptical_arbitrary
    <https://support.lumerical.com/hc/en-us/articles/1500000306621>
    <https://support.lumerical.com/hc/en-us/articles/360042800573>

    Here are the simulation settings used in lumerical

        n_bg=1.44401 #Refractive index of the background material (cladding)
        wg=3.47668   # Refractive index of the waveguide material (core)
        lambda0=1550e-9
        bandwidth = 0e-9
        polarization = 'TE'
        wg_width=500e-9 # Waveguide width
        wg_height=220e-9 # Waveguide height
        etch_depth=80e-9 # etch depth
        theta_fib_mat = 5 # Angle of the fiber mode in material
        theta_taper=30
        efficiency=0.55 # 5.2 dB

    Args:
        parameters: xinput, gap1, width1, gap2, width2 ...
        layer_slab: for slab.
        taper_angle: in deg.
        taper_length: in um.
        fiber_angle: used to compute ellipticity.
        info: optional simulation settings.
        bias_gap: gap/trenches bias (um) to compensate for etching bias.

    Keyword Args:
        taper_length: taper length from input in um.
        taper_angle: grating flare angle in degrees.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        neff: tooth effective index.
        nclad: cladding effective index.
        polarization: te or tm.
        spiked: grating teeth include sharp spikes to avoid non-manhattan drc errors.
        cross_section: cross_section spec for waveguide port.
    """
    return cf.grating_coupler_elliptical_lumerical(
        parameters=parameters,
        layer_slab=layer_slab,
        taper_angle=taper_angle,
        taper_length=taper_length,
        fiber_angle=fiber_angle,
        info=info,
        bias_gap=bias_gap,
        cross_section=cross_section,
    )


grating_coupler_elliptical_lumerical_etch70 = CellAlias(
    grating_coupler_elliptical_lumerical,
    info=dict(
        etch_depth=80e-3,
        link="https://support.lumerical.com/hc/en-us/articles/1500000306621",
        fiber_angle=5,
        width_min=0.1,
        gap_min=0.1,
        efficiency=0.55,
    ),
)
