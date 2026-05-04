"""This module contains the building blocks for the CSPDK PDK."""

from functools import partial

import gdsfactory as gf

from mypdk.tech import LAYER

##############################
# grating couplers Rectangular
##############################


@gf.cell
def grating_coupler_rectangular(
    period=0.315 * 2,
    n_periods: int = 60,
    length_taper: float = 350.0,
    wavelength: float = 1.55,
    cross_section="strip",
) -> gf.Component:
    """A grating coupler with straight and parallel teeth.

    Args:
        period: the period of the grating
        n_periods: the number of grating teeth
        length_taper: the length of the taper tapering up to the grating
        wavelength: the center wavelength for which the grating is designed
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_rectangular(
        n_periods=n_periods,
        period=period,
        fill_factor=0.5,
        width_grating=11.0,
        length_taper=length_taper,
        polarization="te",
        wavelength=wavelength,
        layer_slab=LAYER.WG,
        layer_grating=LAYER.GRA,
        fiber_angle=10.0,
        slab_xmin=-1.0,
        slab_offset=0.0,
        cross_section=cross_section,
    )


grating_coupler_rectangular_rib = partial(
    grating_coupler_rectangular,
    period=0.5,
    cross_section="rib",
    n_periods=60,
)

##############################
# grating couplers elliptical
##############################


@gf.cell
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.315,
    cross_section="strip",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    Args:
        wavelength: the center wavelength for which the grating is designed
        grating_line_width: the line width of the grating
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.0,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        fiber_angle=15.0,
        neff=2.638,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section=cross_section,
    )
