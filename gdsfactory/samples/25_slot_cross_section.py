"""Small demonstration of the slot cross_section utilizing add_center_section=False."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    wg1 = gf.components.straight(length=10, width=0.8, cross_section="slot")
    wg1.show(show_ports=True)  # show it in klayout
