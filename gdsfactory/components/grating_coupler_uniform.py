from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.taper import taper
from gdsfactory.snap import snap_to_grid


@gf.cell
def grating_coupler_uniform(
    num_teeth: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    width: float = 0.5,
    partial_etch: bool = False,
    layer: Tuple[int, int] = gf.LAYER.WG,
    layer_partial_etch: Tuple[int, int] = gf.LAYER.SLAB150,
    polarization: str = "te",
    wavelength: int = 1500,
) -> Component:
    r"""Grating coupler uniform (grating with rectangular shape not elliptical).
    Therefore it needs a longer taper.
    Grating teeth are straight instead of elliptical.

    Args:
        num_teeth: 20
        period: 0.75
        fill_factor: 0.5
        width_grating: 11
        length_taper: 150
        width: 0.5
        partial_etch: False

    .. code::

                 \  \  \  \
                  \  \  \  \
                _|-|_|-|_|-|___
               |_______________  W0

    """
    G = Component()

    if partial_etch:
        partetch_overhang = 5
        teeth = gf.snap.snap_to_grid(period * (1 - fill_factor))
        _compass = compass(
            size=[teeth, width_grating + partetch_overhang * 2],
            layer=layer_partial_etch,
        )

        # make the etched areas (opposite to teeth)
        for i in range(num_teeth):
            cgrating = G.add_ref(_compass)
            dx = gf.snap.snap_to_grid(i * period)
            cgrating.x += dx
            cgrating.y = 0

        # draw the deep etched square around the grating

        xsize = gf.snap.snap_to_grid(num_teeth * period, 2)
        deepbox = G.add_ref(compass(size=[xsize, width_grating], layer=layer))
        deepbox.movex(xsize / 2)
    else:
        for i in range(num_teeth):
            xsize = gf.snap.snap_to_grid(period * fill_factor)
            dx = gf.snap.snap_to_grid(i * period)
            cgrating = G.add_ref(compass(size=[xsize, width_grating], layer=layer))
            cgrating.x += dx
            cgrating.y = 0
    # make the taper
    tgrating = G.add_ref(
        taper(
            length=length_taper,
            width1=width_grating,
            width2=width,
            port=None,
            layer=layer,
        )
    )
    tgrating.xmin = snap_to_grid(cgrating.xmax)
    G.add_port(port=tgrating.ports["o2"], name="o1")
    G.polarization = polarization
    G.wavelength = wavelength
    G.rotate(180)
    gf.asserts.grating_coupler(G)
    return G


if __name__ == "__main__":
    # c = grating_coupler_uniform(name='gcu', partial_etch=True)
    c = grating_coupler_uniform(partial_etch=False)
    print(c.ports)
    c.show()
