"""Auto-taper from silicon nitride to strip cross-section."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.routing.auto_taper import auto_taper_to_cross_section


@gf.cell
def silicon_nitride_strip(width_nitride: float = 1) -> gf.Component:
    c = gf.Component()
    ref = c << gf.c.straight(
        cross_section=gf.cross_section.nitride, width=width_nitride
    )
    port1 = auto_taper_to_cross_section(
        c, port=ref["o1"], cross_section=gf.cross_section.strip
    )
    c.add_port(name="o1", port=port1)
    c.add_port(name="o2", port=ref["o2"])
    return c


if __name__ == "__main__":
    c = silicon_nitride_strip(width_nitride=1)
    c.show()
