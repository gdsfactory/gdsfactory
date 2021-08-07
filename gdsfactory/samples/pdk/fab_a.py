"""Lets for example customize the default gdsfactory PDK

Fab A is mostly defined using Metal layers in GDS layer (30, 0)

The metal layer traces are 2um wide

"""
import dataclasses

import gdsfactory
from gdsfactory.difftest import difftest
from gdsfactory.tech import TECH, Layer, Waveguide


@dataclasses.dataclass
class Metal1(Waveguide):
    width: float = 2.0
    width_wide: float = 10.0
    auto_widen: bool = False
    layer: Layer = (30, 0)
    radius: float = 10.0


METAL1 = Metal1()

TECH.waveguide.metal1 = METAL1


def test_waveguide():
    c = gdsfactory.components.straight(waveguide="metal1")
    difftest(c)


if __name__ == "__main__":

    wg = gdsfactory.components.straight(length=20, waveguide="metal1")
    gc = gdsfactory.components.grating_coupler_elliptical_te(
        layer=METAL1.layer, wg_width=METAL1.width
    )

    wg_gc = gdsfactory.routing.add_fiber_array(
        component=wg, grating_coupler=gc, waveguide="metal1"
    )
    wg_gc.show()
