"""Lets for example customize the default gdsfactory PDK

Fab B is mostly uses optical layers but the waveguides required many cladding layers to avoid tiling, dopants...

Lets say that the waveguides are defined in layer (2, 0) and are 0.3um wide

"""
from typing import Tuple

import pydantic.dataclasses as dataclasses

from pp.difftest import difftest
from pp.tech import TECH, Layer, Waveguide


@dataclasses.dataclass
class StripB(Waveguide):
    width: float = 0.3
    layer: Layer = (2, 0)
    width_wide: float = 10.0
    auto_widen: bool = False
    radius: float = 10.0
    layers_cladding: Tuple[Layer, ...] = ((71, 0), (68, 0))


STRIPB = StripB()

# register the new waveguide dynamically
TECH.waveguide.stripb = STRIPB


def test_waveguide():
    import pp

    wg = pp.components.straight(length=20, waveguide="stripb")
    gc = pp.components.grating_coupler_elliptical_te(
        layer=STRIPB.layer, wg_width=STRIPB.width
    )

    wg_gc = pp.routing.add_fiber_array(
        component=wg, grating_coupler=gc, waveguide="stripb"
    )
    wg_gc.show()
    difftest(wg_gc)


if __name__ == "__main__":
    import pp

    c = pp.components.mmi2x2(layer=STRIPB.layer)
    gc = pp.components.grating_coupler_elliptical_te(
        layer=STRIPB.layer, wg_width=STRIPB.width
    )

    c_gc = pp.routing.add_fiber_array(
        component=c, grating_coupler=gc, waveguide="stripb"
    )
    c_gc.show()
