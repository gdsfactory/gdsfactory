from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import Delta

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@gf.cell
def alphabet(dx: Delta = 10) -> gf.Component:
    c = gf.Component()
    x = 0.0
    for s in characters:
        ci = gf.components.text(text=s)
        ci.name = s
        ci.flatten()
        char = c << ci
        char.dx = x
        x += dx

    return c


if __name__ == "__main__":
    c = alphabet()
    c.write_gds("alphabet.gds")
    c.show()
