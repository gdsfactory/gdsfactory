from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper_strip_to_slab150
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec

_gaps = (0.2,) * 10
_widths = (0.5,) * 10


@gf.cell
def grating_coupler_rectangular_arbitrary_slab(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = taper_strip_to_slab150,
    layer_slab: LayerSpec | None = "SLAB150",
    layer_grating: LayerSpec | None = None,
    slab_offset: float = 2.0,
    fiber_angle: float = 15,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Grating coupler uniform (grating with rectangular shape not elliptical). Therefore it needs a longer taper. Grating teeth are straight instead of elliptical.

    Args:
        gaps: list of gaps.
        widths: list of widths.
        width_grating: um.
        length_taper: um.
        polarization: 'te' or 'tm'.
        wavelength: in um.
        taper: function.
        layer_slab: for pedestal.
        layer_grating: for grating.
        slab_offset: from edge.
        fiber_angle: in degrees.
        cross_section: for input waveguide port.
        kwargs: cross_section settings.

    .. code::

        side view
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
    layer = layer_grating or xs.layer

    c = Component()
    taper = gf.get_component(
        taper,
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        w_slab2=width_grating + 2 * slab_offset,
    )

    taper_ref = c << taper

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    x0 = xi = taper_ref.dxmax

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
        cgrating.dx = gf.snap.snap_to_grid(xi)
        cgrating.dy = 0
        xi += width / 2

    if layer_slab:
        slab = c << rectangle(
            size=(
                gf.snap.snap_to_grid(xi - x0) + slab_offset,
                width_grating + 2 * slab_offset,
            ),
            layer=layer_slab,
            port_type=None,
            centered=True,
        )
        slab.dxmin = x0

    xport = np.round((xi + length_taper) / 2, 3)

    c.add_port(
        name="o2",
        port_type=f"vertical_{polarization}",
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=layer,
    )
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = fiber_angle
    xs.add_bbox(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_rectangular_arbitrary_slab(slab_offset=2.0)
    print(c.ports)
    c.show()
