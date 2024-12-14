from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_couplers.functions import (
    grating_taper_points,
    grating_tooth_points,
)
from gdsfactory.functions import DEG2RAD
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@gf.cell
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
    layer = xs.layer
    assert layer is not None

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff**2 - nclad**2 * sthc**2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * nclad * sthc / d

    a1 = float(round(a1, 3))
    b1 = float(round(b1, 3))
    x1 = float(round(x1, 3))

    period = a1 + x1

    c = gf.Component()
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    # Make the taper
    p = taper_length / period
    a_taper = a1 * p
    b_taper = b1 * p
    x_taper = x1 * p

    x_output = a_taper + x_taper - taper_length
    pts = grating_taper_points(
        a=a_taper,
        b=b_taper,
        x0=x_output,
        taper_length=x_taper,
        taper_angle=taper_angle,
        wg_width=wg_width,
    )
    c.add_polygon(pts, layer)

    width = gf.snap.snap_to_grid(grating_line_width)
    gap = gf.snap.snap_to_grid(period - grating_line_width)

    xi = taper_length
    for p in range(n_periods):
        xi += gap + width / 2
        p = xi / period
        pts = grating_tooth_points(
            p * a1, p * b1, p * x1, width, taper_angle, spiked=spiked
        )
        c.add_polygon(pts, layer)
        xi += width / 2

    w = 1.0
    total_length = (
        period * n_periods
        + taper_length
        + grating_line_width / 2
        + period
        - grating_line_width
        + w / 2
    )

    if big_last_tooth:
        # Add last "large tooth" after the standard grating teeth
        a = total_length / (1 + x1 / a1)
        b = b1 / a1 * a
        x = x1 / a1 * a

        pts = grating_tooth_points(a, b, x, w, taper_angle, spiked=False)
        c.add_polygon(pts, layer)

    x = np.round(taper_length + x_output, 3)

    c.add_port(
        name="o1",
        center=(x_output, 0),
        width=wg_width,
        orientation=180,
        layer=layer,
        port_type="optical",
    )

    if layer_slab:
        slab_xmin += x_output + taper_length
        slab_length = total_length + slab_offset
        slab_width = (c.dysize + 2 * slab_offset) / 2
        c.add_polygon(
            [
                (slab_xmin, slab_width),
                (slab_length, slab_width),
                (slab_length, -slab_width),
                (slab_xmin, -slab_width),
            ],
            layer_slab,
        )

    xs.add_bbox(c)
    c.add_port(
        name="o2",
        center=(x, 0),
        width=10,
        orientation=0,
        layer=layer,
        port_type=f"vertical_{polarization}",
    )
    return c


grating_coupler_elliptical_tm = partial(
    grating_coupler_elliptical,
    grating_line_width=0.707,
    polarization="tm",
    taper_length=30,
    slab_xmin=-2,
    neff=1.8,
    n_periods=16,
)


grating_coupler_elliptical_te = grating_coupler_elliptical


if __name__ == "__main__":
    # c = grating_coupler_elliptical_tm(taper_length=30)
    # c = grating_coupler_elliptical_te(cladding_layers=((2, 0), (3, 0)))
    # c = grating_coupler_elliptical(layer=(2, 0), taper_length=50, slab_xmin=-5)
    # print(c.polarization)
    # print(c.wavelength)
    # print(c.ports)
    # c.pprint()
    # c = gf.c.extend_ports(c)
    # c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_elliptical, with_loopback=False)

    # c = gf.components.grating_coupler_elliptical_te()
    c = gf.components.grating_coupler_elliptical_tm()
    c.show()
