from __future__ import annotations

from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_couplers.grating_coupler_elliptical_arbitrary import (
    grating_coupler_elliptical_arbitrary,
)
from gdsfactory.typings import CrossSectionSpec, Floats, LayerSpec

parameters = (
    -2.4298362615732447,
    0.1,
    0.48007023217536954,
    0.1,
    0.607397685752365,
    0.1,
    0.4498844003086115,
    0.1,
    0.4274116312627637,
    0.1,
    0.4757904248387285,
    0.1,
    0.5026649898504233,
    0.10002922416240886,
    0.5100366774007897,
    0.1,
    0.494399635363353,
    0.1079599958465788,
    0.47400592737426483,
    0.14972685326277918,
    0.43272750134545823,
    0.1839530796530385,
    0.3872023336708212,
    0.2360175325711591,
    0.36032212454768675,
    0.24261846353500535,
    0.35770350120764394,
    0.2606637836858316,
    0.3526104381544335,
    0.24668202254540886,
    0.3717488388788273,
    0.22920754299702897,
    0.37769616507688464,
    0.2246528336925301,
    0.3765437598650894,
    0.22041773376471022,
    0.38047596041838994,
    0.21923601658169187,
    0.3798873698864591,
    0.21700438236445285,
    0.38291698672245644,
    0.21827768053295463,
    0.3641322152037017,
    0.23729077006065105,
    0.3676834419346081,
    0.24865079519725933,
    0.34415050295044936,
    0.2733570818755685,
    0.3306230780901629,
    0.27350446437732157,
)


@gf.cell
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
    https://support.lumerical.com/hc/en-us/articles/1500000306621
    https://support.lumerical.com/hc/en-us/articles/360042800573

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
        layer: for waveguide.
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
    parameters = tuple(parameters)
    xinput = parameters[0]
    teeth_list = parameters[1:]
    gaps = teeth_list[::2]
    widths = teeth_list[1::2]
    info = info or {}
    gaps = tuple(gap + bias_gap for gap in gaps)

    c = grating_coupler_elliptical_arbitrary(
        gaps=gaps,
        widths=widths,
        taper_angle=taper_angle,
        taper_length=taper_length,
        layer_slab=layer_slab,
        fiber_angle=fiber_angle,
        cross_section=cross_section,
    )
    c.info.update(info)
    c.info["xinput"] = xinput
    return c


grating_coupler_elliptical_lumerical_etch70 = partial(
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

if __name__ == "__main__":
    c = grating_coupler_elliptical_lumerical_etch70()
    # c = grating_coupler_elliptical_lumerical()
    c.show()
