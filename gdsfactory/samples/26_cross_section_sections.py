"""Small demonstration of the cross_section."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = gf.c.straight(cross_section="rib")
    p = c["o1"]
    xs = p.cross_section
    # Sections are keyed by LayerInfo objects, not their string representations.
    s = xs.sections[gf.get_layer_info("SLAB90")]
