from typing import Tuple
from pp.components import taper
from pp.components import compass
from pp.name import autoname
from pp.component import Component
from pp.components.grating_coupler import grating_coupler
import pp


@grating_coupler
@autoname
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
    polarization="te",
    wavelength=1500,
) -> Component:
    """Grating coupler uniform

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

    """
    # returns a fiber grating
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
    return G


if __name__ == "__main__":
    # c = grating_coupler_uniform(name='gcu', partial_etch=True)
    c = grating_coupler_uniform(partial_etch=False, pins=True)
    pp.show(c)
