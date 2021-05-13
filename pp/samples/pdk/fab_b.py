"""Lets for example customize the default gdsfactory PDK

Fab B is mostly uses optical layers but the waveguides required many cladding layers to avoid tiling, dopants...

Lets say that the waveguides are defined in layer (2, 0) and are 0.3um wide

"""
import dataclasses
from typing import Tuple

from pp.tech import TECH, Layer, Mmi1x2, Waveguide


@dataclasses.dataclass
class StripB(Waveguide):
    width: float = 0.3
    width_wide: float = 10.0
    auto_widen: bool = False
    layer: Layer = (2, 0)
    radius: float = 10.0
    min_spacing: float = 10.0
    layers_cladding: Tuple[Layer] = ((71, 0), (68, 0))


STRIPB = StripB()

# register the new waveguide dynamically
TECH.waveguide.stripb = STRIPB


class Mmi1x2FabB(Mmi1x2):
    width: float = 0.3
    width_taper: float = 0.6
    length_taper: float = 6.0
    length_mmi: float = 3.5
    width_mmi: float = 2.0
    gap_mmi: float = 0.25


# lets register an MMI for this fab
TECH.components.mmi1x2b = Mmi1x2FabB()


def demo_waveguide():
    import pp

    wg = pp.components.straight(length=20, cross_section_name="stripb")
    gc = pp.components.grating_coupler_elliptical_te(
        layer=STRIPB.layer, wg_width=STRIPB.width
    )

    wg_gc = pp.routing.add_fiber_array(
        component=wg, grating_coupler=gc, cross_section_name="stripb"
    )
    wg_gc.show()


if __name__ == "__main__":
    import pp

    wg = pp.components.factory()
    gc = pp.components.grating_coupler_elliptical_te(
        layer=STRIPB.layer, wg_width=STRIPB.width
    )

    wg_gc = pp.routing.add_fiber_array(
        component=wg, grating_coupler=gc, cross_section_name="stripb"
    )
    wg_gc.show()
