from __future__ import annotations

from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.typings import CrossSectionSpec, LayerSpecs


@gf.cell
def terminator(
    length: Optional[float] = 50,
    cross_section_input: CrossSectionSpec = "strip",
    cross_section_tip: Optional[CrossSectionSpec] = None,
    tapered_width: float = 0.2,
    doping_layers: LayerSpecs = ("NPP",),
) -> gf.Component:
    """Returns doped taper to terminate waveguides.

    Args:
        length: distance between input and narrow tapered end.
        cross_section_input: input cross-section.
        cross_section_tip: cross-section at the end of the termination.
        tapered_width: width of the default cross-section at the end of the termination.
            Only used if cross_section_tip is not None.
        doping_layers: doping layers to superimpose on the taper. Default N++.
    """
    c = Component()

    cross_section_tip = cross_section_tip or gf.get_cross_section(
        cross_section_input, width=tapered_width
    )

    taper = c << gf.get_component(
        taper_cross_section,
        length=length,
        cross_section1=cross_section_input,
        cross_section2=cross_section_tip,
    )

    for layer in doping_layers:
        c << gf.components.bbox(bbox=taper.bbox, layer=layer)

    c.add_port(name="o1", port=taper.ports["o1"])
    return c


if __name__ == "__main__":
    c = terminator(cross_section_input=gf.partial(gf.cross_section.strip, width=10))
    c.show(show_ports=True)
