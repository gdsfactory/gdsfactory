from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical import (
    grating_taper_points,
    grating_tooth_points,
)
from gdsfactory.geometry.functions import DEG2RAD
from gdsfactory.tech import LAYER
from gdsfactory.types import Floats, Layer

_gaps = (0.1,) * 10
_widths = (0.5,) * 10


@gf.cell
def grating_coupler_elliptical_arbitrary(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    wg_width: float = 0.5,
    taper_length: float = 16.6,
    taper_angle: float = 60.0,
    layer: Tuple[int, int] = LAYER.WG,
    wavelength: float = 1.554,
    fiber_angle: float = 15.0,
    neff: float = 2.638,  # tooth effective index
    nclad: float = 1.443,
    layer_slab: Optional[Tuple[int, int]] = LAYER.SLAB150,
    slab_xmin: float = -3.0,
    polarization: str = "te",
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Optional[Layer] = gf.LAYER.TE,
    spiked: bool = True,
) -> Component:
    r"""Grating coupler with parametrization based on Lumerical FDTD simulation.

    The ellipticity is derived from Lumerical knowdledge base
    it depends on fiber_angle (degrees), neff, and nclad

    Args:
        gaps:
        widths:
        wg_width: waveguide width
        taper_length: taper length from input
        taper_angle: grating flare angle
        layer: LAYER.WG
        wavelength: grating transmission central wavelength (um)
        fiber_angle: fibre angle in degrees determines ellipticity
        neff: tooth effective index
        nclad: cladding effective index
        layer_slab: Optional slab
        slab_xmin: where 0 is at the start of the taper
        polarization: te or tm
        fiber_marker_width
        fiber_marker_layer
        spiked: grating teeth have sharp spikes to avoid non-manhattan drc errors


    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|

    """

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * DEG2RAD)
    d = neff ** 2 - nclad ** 2 * sthc ** 2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * nclad * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)
    period = a1 + x1

    # https://en.wikipedia.org/wiki/Ellipse
    c = (a1 ** 2 - b1 ** 2) ** 0.5

    # e = (1 - (b1 / a1) ** 2) ** 0.5
    # print(e)

    c = gf.Component()
    c.info.polarization = polarization
    c.info.wavelength = wavelength

    gaps = gf.snap.snap_to_grid(gaps)
    widths = gf.snap.snap_to_grid(widths)

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
        midpoint=[x, 0],
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
        port_type=name,
    )

    # Add port
    c.add_port(
        name="o1", midpoint=[x_output, 0], width=wg_width, orientation=180, layer=layer
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
    return c


if __name__ == "__main__":
    c = grating_coupler_elliptical_arbitrary(fiber_angle=8)
    c.show()
