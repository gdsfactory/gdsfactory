from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.tapers.taper import taper
from gdsfactory.typings import CrossSectionSpec, Floats, LayerSpec

_gaps = (0.2,) * 10
_widths = (0.5,) * 10


@gf.cell
def grating_coupler_rectangular_arbitrary(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    layer_grating: LayerSpec | None = None,
    layer_slab: LayerSpec | None = None,
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
    fiber_angle: float = 15,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Grating coupler uniform with rectangular shape (not elliptical).

    Therefore it needs a longer taper.
    Grating teeth are straight instead of elliptical.

    Args:
        gaps: list of gaps between grating teeth.
        widths: list of grating widths.
        width_grating: grating teeth width.
        length_taper: taper length (um).
        polarization: 'te' or 'tm'.
        wavelength: in um.
        layer_grating: Optional layer for grating. \
                by default None uses cross_section.layer. \
                if different from cross_section.layer expands taper.
        layer_slab: layer that protects the slab under the grating.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
        fiber_angle: in degrees.
        cross_section: for input waveguide port.

    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|



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

    """
    xs = gf.get_cross_section(cross_section)
    assert xs.layer is not None
    layer_wg = gf.get_layer(xs.layer)
    layer_grating = layer_grating or layer_wg

    layer_grating = gf.get_layer(layer_grating)
    c = Component()

    taper_ref = c << taper(
        length=length_taper,
        width1=xs.width,
        width2=width_grating,
        cross_section=cross_section,
    )

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    xi = length_taper

    y0 = width_grating / 2

    for width, gap in zip(widths, gaps):
        xi += gap
        points = np.array(
            [
                [xi, -y0],
                [xi, +y0],
                [xi + width, +y0],
                [xi + width, -y0],
            ]
        )
        c.add_polygon(
            points,
            layer_grating,
        )
        xi += width

    if layer_slab:
        slab_xmin = length_taper - slab_offset
        slab_xmax = length_taper + np.sum(widths) + np.sum(gaps) + slab_offset
        slab_ysize = width_grating + 2 * slab_offset
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
    xport = np.round((xi + length_taper) / 2, 3)
    c.add_port(
        name="o2",
        port_type=f"vertical_{polarization}",
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=xs.layer,
    )
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = fiber_angle

    xs.add_bbox(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_rectangular_arbitrary(
        cross_section="rib_bbox", slab_offset=2.0, layer_slab=(2, 0)
    )
    c.show()
