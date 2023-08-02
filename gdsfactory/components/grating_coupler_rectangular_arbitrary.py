from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec

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
    taper: ComponentSpec | None = taper_function,
    layer_grating: LayerSpec | None = None,
    layer_slab: LayerSpec = "SLAB150",
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
    fiber_angle: float = 15,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Grating coupler uniform with rectangular shape (not elliptical). Therefore it needs a longer taper. Grating teeth are straight instead of elliptical.

    Args:
        gaps: list of gaps between grating teeth.
        widths: list of grating widths.
        width_grating: grating teeth width.
        length_taper: taper length (um).
        polarization: 'te' or 'tm'.
        wavelength: in um.
        taper: function.
        layer_grating: Optional layer for grating.
            by default None uses cross_section.layer.
            if different from cross_section.layer expands taper.
        layer_slab: layer that protects the slab under the grating.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
        fiber_angle: in degrees.
        cross_section: for input waveguide port.
        kwargs: cross_section settings.

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
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer_wg = gf.get_layer(xs.layer)
    layer_grating = gf.get_layer(layer_grating) or layer_wg
    c = Component()

    if taper:
        taper_ref = c << gf.get_component(
            taper,
            length=length_taper,
            width2=width_grating,
            width1=wg_width,
            layer=xs.layer,
        )

        c.add_port(port=taper_ref.ports["o1"], name="o1")
        xi = taper_ref.xmax
    else:
        length_taper = 0
        xi = 0

    widths = gf.snap.snap_to_grid(widths)
    gaps = gf.snap.snap_to_grid(gaps)

    y0 = width_grating / 2

    for width, gap in zip(widths, gaps):
        xi += gap
        points = snap_to_grid(
            np.array(
                [
                    [xi, -y0],
                    [xi, +y0],
                    [xi + width, +y0],
                    [xi + width, -y0],
                ]
            )
        )
        c.add_polygon(
            points,
            layer_grating,
        )
        xi += width

    if layer_slab:
        slab_xmin = length_taper - slab_offset
        slab_xmax = length_taper + np.sum(widths) + np.sum(gaps) + slab_offset
        slab_ysize = c.ysize + 2 * slab_offset
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
    name = f"opt_{polarization.lower()}_{int(wavelength*1e3)}_{int(fiber_angle)}"
    c.add_port(
        name=name,
        port_type=name,
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=xs.layer,
    )
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength

    gf.asserts.grating_coupler(c)
    if xs.add_bbox:
        c = xs.add_bbox(c)
    if xs.add_pins:
        c = xs.add_pins(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_rectangular_arbitrary()
    # c = grating_coupler_rectangular_arbitrary(
    #     layer_grating=(3, 0), layer_slab=(2, 0), slab_offset=1
    # )
    c.show(show_ports=True)
