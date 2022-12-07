from __future__ import annotations

from typing import List, Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import CrossSection, strip
from gdsfactory.types import LayerSpec


@gf.cell
def terminator(
    length: Optional[float] = 50,
    input_xs: Optional[CrossSection] = strip,
    tapered_xs: Optional[CrossSection] = None,
    doping_layers: List[LayerSpec] = ["NPP"],
    **kwargs,
) -> gf.Component:
    """Doped narrow taper to terminate waveguides.

    Args:
        length: distance between input and narrow tapered end.
        input_xs: input cross-section.
        tapered_xs: cross-section at the end of the termination (by default, input_xs with width 200 nm)
        doping_layers: doping layers to superimpose on the taper. Default N++.
        **kwargs: taper arguments
    """
    c = Component()

    tapered_xs = tapered_xs if tapered_xs else gf.partial(input_xs, width=0.2)

    taper = c << gf.get_component(
        taper_cross_section,
        length=length,
        cross_section1=input_xs,
        cross_section2=tapered_xs,
    )

    for layer in doping_layers:
        c << gf.components.bbox(bbox=taper.bbox, layer=layer)

    c.add_port(name="o1", port=taper.ports["o1"])

    return c


if __name__ == "__main__":

    c = terminator()
    c.show(show_ports=True)
