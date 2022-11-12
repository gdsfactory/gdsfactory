import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical import (
    grating_taper_points,
    grating_tooth_points,
)
from gdsfactory.geometry.functions import DEG2RAD
from gdsfactory.types import CrossSectionSpec, Floats, LayerSpec

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
    neff: float = 2.638,  # tooth effective index
    nclad: float = 1.443,
    layer_slab: LayerSpec = "SLAB150",
    slab_xmin: float = -3.0,
    polarization: str = "te",
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: LayerSpec = "TE",
    spiked: bool = True,
    bias_gap: float = 0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    The ellipticity is derived from Lumerical knowdledge base
    it depends on fiber_angle (degrees), neff, and nclad

    Args:
        gaps: list of gaps.
        widths: list of widths.
        taper_length: taper length from input.
        taper_angle: grating flare angle.
        wavelength: grating transmission central wavelength (um).
        fiber_angle: fibre angle in degrees determines ellipticity.
        neff: tooth effective index to compute ellipticity.
        nclad: cladding effective index to compute ellipticity.
        layer_slab: Optional slab.
        slab_xmin: where 0 is at the start of the taper.
        polarization: te or tm.
        fiber_marker_width: in um.
        fiber_marker_layer: Optional marker.
        spiked: grating teeth have spikes to avoid drc errors..
        bias_gap: etch gap (um).
            Positive bias increases gap and reduces width to keep period constant.
        cross_section: cross_section spec for waveguide port.
        kwargs: cross_section settings.

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
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer = xs.layer

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff**2 - nclad**2 * sthc**2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * nclad * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)
    period = a1 + x1

    c = gf.Component()
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    gaps = gf.snap.snap_to_grid(np.array(gaps) + bias_gap)
    widths = gf.snap.snap_to_grid(np.array(widths) - bias_gap)

    xi = taper_length
    for gap, width in zip(gaps, widths):
        xi += gap + width / 2
        p = xi / period
        pts = grating_tooth_points(
            p * a1, p * b1, p * x1, width, taper_angle, spiked=spiked
        )
        c.add_polygon(pts, layer)
        xi += width / 2

    # Make the taper
    p = taper_length / period
    a_taper = p * a1
    b_taper = p * b1
    x_taper = p * x1

    x_output = a_taper + x_taper - taper_length + widths[0] / 2
    pts = grating_taper_points(
        a_taper, b_taper, x_output, x_taper, taper_angle, wg_width=wg_width
    )
    c.add_polygon(pts, layer)
    x = (taper_length + xi) / 2
    name = f"vertical_{polarization.lower()}"
    c.add_port(
        name=name,
        center=(x, 0),
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
        port_type=name,
    )

    # Add port
    c.add_port(
        name="o1",
        center=(x_output, 0),
        width=wg_width,
        orientation=180,
        layer=layer,
        cross_section=xs,
    )

    if layer_slab:
        slab_xmin += taper_length
        slab_xsize = xi + 2.0
        slab_ysize = c.ysize + 2.0
        yslab = slab_ysize / 2
        c.add_polygon(
            [
                (slab_xmin, yslab),
                (slab_xsize, yslab),
                (slab_xsize, -yslab),
                (slab_xmin, -yslab),
            ],
            layer_slab,
        )

    if fiber_marker_layer:
        circle = gf.components.circle(
            radius=fiber_marker_width / 2, layer=fiber_marker_layer
        )
        circle_ref = c.add_ref(circle)
        circle_ref.movex(x)
    if xs.add_bbox:
        c = xs.add_bbox(c)
    if xs.add_pins:
        c = xs.add_pins(c)
    return c


@gf.cell
def grating_coupler_elliptical_uniform(
    n_periods: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    **kwargs,
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    The ellipticity is derived from Lumerical knowdledge base
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
        slab_xmin: where 0 is at the start of the taper.
        polarization: te or tm.
        fiber_marker_width: in um.
        fiber_marker_layer: Optional marker.
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
    widths = [period * fill_factor] * n_periods
    gaps = [period * (1 - fill_factor)] * n_periods
    return grating_coupler_elliptical_arbitrary(gaps=gaps, widths=widths, **kwargs)


if __name__ == "__main__":
    # c = grating_coupler_elliptical_arbitrary()
    c = grating_coupler_elliptical_uniform(n_periods=3, fill_factor=0.1)
    # c = grating_coupler_elliptical_arbitrary(fiber_angle=8, bias_gap=-0.05)
    # c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_elliptical_arbitrary)
    c.show(show_ports=True)
