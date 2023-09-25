from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.typings import CrossSectionSpec


@gf.cell
def straight_array(
    n: int = 4,
    spacing: float = 4.0,
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> Component:
    """Array of straights connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n: number of straights.
        spacing: edge to edge straight spacing.
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
    """
    c = Component()
    wg = straight(cross_section=cross_section, length=length)

    for i in range(n):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.info["width"])
        c.add_ports(wref.ports, prefix=str(i))

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = straight_array()
    # c.pprint_ports()
    c.show(show_ports=True)
