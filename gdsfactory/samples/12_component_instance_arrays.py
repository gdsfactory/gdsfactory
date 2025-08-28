"""Lets access the ports for an array of instances."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    aref = c.add_ref(gf.components.straight_array(), columns=3, column_pitch=20)
    print(aref.ports["o1", 0, 0].x)
    print(aref.ports["o1", 1, 0].x)
    print(aref.ports["o1", 2, 0].x)
    c.show()
