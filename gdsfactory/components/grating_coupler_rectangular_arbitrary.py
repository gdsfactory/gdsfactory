from typing import Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Floats, LayerSpec

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
    taper: Optional[ComponentSpec] = taper_function,
    layer_slab: LayerSpec = "SLAB150",
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
    fiber_marker_layer: LayerSpec = "TE",
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
        layer_slab: layer that protects the slab under the grating.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
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
    layer = xs.layer

    c = Component()

    if taper:
        taper_ref = c << gf.get_component(
            taper,
            length=length_taper,
            width2=width_grating,
            width1=wg_width,
            layer=layer,
        )

        c.add_port(port=taper_ref.ports["o1"], name="o1")
        xi = taper_ref.xmax
    else:
        length_taper = 0
        xi = 0

    widths = gf.snap.snap_to_grid(widths)
    gaps = gf.snap.snap_to_grid(gaps)

    for width, gap in zip(widths, gaps):
        xi += gap + width / 2
        cgrating = c.add_ref(
            rectangle(
                size=(width, width_grating),
                layer=layer,
                port_type=None,
                centered=True,
            )
        )
        cgrating.x = gf.snap.snap_to_grid(xi)
        cgrating.y = 0
        xi += width / 2

    if layer_slab:
        slab_xmin += length_taper
        slab_xsize = xi + slab_offset
        slab_ysize = c.ysize + 2 * slab_offset
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
    xport = np.round((xi + length_taper) / 2, 3)
    port_type = f"vertical_{polarization.lower()}"
    c.add_port(
        name=port_type,
        port_type=port_type,
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=fiber_marker_layer,
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
    print(c.ports)
    c.show(show_ports=True)
