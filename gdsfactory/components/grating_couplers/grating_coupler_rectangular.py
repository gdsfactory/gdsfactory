from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


@gf.cell_with_module_name
def grating_coupler_rectangular(
    n_periods: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = "taper",
    layer_slab: LayerSpec | None = "SLAB150",
    layer_grating: LayerSpec | None = None,
    fiber_angle: float = 15,
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Grating coupler with rectangular shapes (not elliptical).

    Needs longer taper than elliptical.
    Grating teeth are straight.
    For a focusing grating take a look at grating_coupler_elliptical.

    Args:
        n_periods: number of grating teeth.
        period: grating pitch.
        fill_factor: ratio of grating width vs gap.
        width_grating: 11.
        length_taper: 150.
        polarization: 'te' or 'tm'.
        wavelength: in um.
        taper: function.
        layer_slab: layer that protects the slab under the grating.
        layer_grating: layer for the grating.
        fiber_angle: in degrees.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
        cross_section: for input waveguide port.

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
    xs = gf.get_cross_section(cross_section)
    wg_width = xs.width
    layer = layer_grating or xs.layer
    assert layer is not None

    c = Component()
    taper_ref = c << gf.get_component(
        taper,
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        cross_section=cross_section,
    )

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    x0 = length_taper
    for i in range(n_periods):
        xsize = gf.snap.snap_to_grid(period * fill_factor)
        cgrating = c.add_ref(
            gf.c.rectangle(size=(xsize, width_grating), layer=layer, port_type=None)
        )
        cgrating.xmin = gf.snap.snap_to_grid(x0 + i * period)
        cgrating.y = 0

    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = fiber_angle

    if layer_slab:
        slab_xmin = length_taper - slab_offset
        slab_xmax = length_taper + n_periods * period + slab_offset
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
    xs.add_bbox(c)
    xport = np.round((x0 + cgrating.x) / 2, 3)
    c.add_port(
        name="o2",
        port_type=f"vertical_{polarization}",
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=layer,
    )
    c.flatten()
    return c


if __name__ == "__main__":
    # c = grating_coupler_rectangular(name='gcu', partial_etch=True)
    c = grating_coupler_rectangular(cross_section="rib_bbox", slab_offset=4.0)
    # c = gf.routing.add_fiber_array(grating_coupler=grating_coupler_rectangular)
    print(c.ports)
    c.show()
