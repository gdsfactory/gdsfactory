from typing import Tuple

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.compass import compass
from pp.components.taper import taper


@cell
def grating_coupler_uniform(
    num_teeth: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    width: float = 0.5,
    partial_etch: bool = False,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layer_partial_etch: Tuple[int, int] = pp.LAYER.SLAB150,
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

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_uniform()
      pp.plotgds(c)

    .. code::

                 \  \  \  \
                  \  \  \  \
                _|-|_|-|_|-|___
               |_______________  W0

    """
    G = Component()

    if partial_etch:
        partetch_overhang = 5
        _compass = compass(
            size=[period * (1 - fill_factor), width_grating + partetch_overhang * 2],
            layer=layer_partial_etch,
        )

        # make the etched areas (opposite to teeth)
        for i in range(num_teeth):
            cgrating = G.add_ref(_compass)
            cgrating.x += i * period

        # draw the deep etched square around the grating
        deepbox = G.add_ref(
            compass(size=[num_teeth * period, width_grating], layer=layer)
        )
        deepbox.movex(num_teeth * period / 2)
    else:
        for i in range(num_teeth):
            cgrating = G.add_ref(
                compass(size=[period * fill_factor, width_grating], layer=layer)
            )
            cgrating.x += i * period
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
    tgrating.xmin = cgrating.xmax
    G.add_port(port=tgrating.ports["2"], name="W0")
    G.polarization = polarization
    G.wavelength = wavelength
    G.rotate(180)
    pp.assert_grating_coupler_properties(G)
    return G


if __name__ == "__main__":
    # c = grating_coupler_uniform(name='gcu', partial_etch=True)
    c = grating_coupler_uniform(partial_etch=False)
    print(c.ports)
    pp.show(c)
