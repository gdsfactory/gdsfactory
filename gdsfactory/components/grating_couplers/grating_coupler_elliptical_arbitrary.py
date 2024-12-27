from __future__ import annotations

from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_couplers.functions import (
    grating_taper_points,
    grating_tooth_points,
)
from gdsfactory.functions import DEG2RAD
from gdsfactory.typings import CrossSectionSpec, Floats, LayerSpec

_gaps = (0.1,) * 10
_widths = (0.5,) * 10


@gf.cell
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

    https://en.wikipedia.org/wiki/Ellipse
    c = (a1 ** 2 - b1 ** 2) ** 0.5
    e = (1 - (b1 / a1) ** 2) ** 0.5
    print(e)

    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|

    """
    xs = gf.get_cross_section(cross_section)
    wg_width = xs.width
    assert xs.layer is not None
    layer_wg = gf.get_layer(xs.layer)

    layer_grating = layer_grating or layer_wg
    layer_grating = gf.get_layer(layer_grating)
    sthc = np.sin(fiber_angle * DEG2RAD)

    # generate component
    c = gf.Component()
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    # get the physical parameters needed to compute ellipses
    gaps_array = gf.snap.snap_to_grid(np.array(gaps) + bias_gap)
    widths_array = gf.snap.snap_to_grid(np.array(widths) - bias_gap)
    periods = [g + w for g, w in zip(gaps_array, widths_array)]
    neffs = [wavelength / p + nclad * sthc for p in periods]
    ds = [neff**2 - nclad**2 * sthc**2 for neff in neffs]
    a1s = [round(wavelength * neff / d, 3) for neff, d in zip(neffs, ds)]
    b1s = [round(wavelength / np.sqrt(d), 3) for d in ds]
    x1s = [round(wavelength * nclad * sthc / d, 3) for d in ds]
    xis = np.add(
        taper_length + np.cumsum(periods), -widths_array / 2
    )  # position of middle of each tooth
    ps = np.divide(xis, periods)

    # grating teeth
    for a1, b1, x1, p, width in zip(a1s, b1s, x1s, ps, widths_array):
        pts = grating_tooth_points(
            p * a1, p * b1, p * x1, float(width), taper_angle, spiked=spiked
        )
        c.add_polygon(pts, layer_grating)

    # taper
    p = taper_length / periods[0]  # (gaps[0]+widths[0])
    a_taper = p * a1s[0]
    b_taper = p * b1s[0]
    x_taper = p * x1s[0]
    x_output = a_taper + x_taper - taper_length + widths_array[0] / 2

    if layer_grating == layer_wg:
        pts = grating_taper_points(
            a_taper, b_taper, x_output, x_taper, taper_angle, wg_width=wg_width
        )
        c.add_polygon(pts, layer_wg)

    else:
        pts = grating_taper_points(
            a_taper,
            b_taper,
            x_output,
            x_taper + np.sum(widths_array) + np.sum(gaps_array) + 1,
            taper_angle,
            wg_width=wg_width,
        )
        c.add_polygon(pts, layer=layer_wg)

    c.add_port(
        name="o1",
        center=(x_output, 0),
        width=wg_width,
        orientation=180,
        layer=layer_wg,
        cross_section=xs,
    )

    if layer_slab:
        slab_xmin = taper_length + taper_to_slab_offset
        slab_xmax = c.dxmax + 0.5
        slab_ysize = c.dysize + 2.0
        yslab = slab_ysize / 2
        c.add_polygon(
            [
                (slab_xmin, yslab),
                (slab_xmax, yslab),
                (slab_xmax, -yslab),
                (slab_xmin, -yslab),
            ],
            layer_slab,
        )

    xs.add_bbox(c)
    x = (taper_length + xis[-1]) / 2
    c.add_port(
        name="o2",
        center=(x, 0),
        width=10,
        orientation=0,
        layer=xs.layer,
        port_type=f"vertical_{polarization}",
    )
    return c


@gf.cell
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

    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|

    """
    widths = (period * fill_factor,) * n_periods
    gaps = (period * (1 - fill_factor),) * n_periods
    return grating_coupler_elliptical_arbitrary(gaps=gaps, widths=widths, **kwargs)


if __name__ == "__main__":
    c = grating_coupler_elliptical_arbitrary(layer_grating=(3, 0))
    # c = grating_coupler_elliptical_arbitrary()
    c.show()
