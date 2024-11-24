from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import Delta

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@gf.cell
def alphabet(dx: Delta = 10) -> gf.Component:
    c = gf.Component()
    x = 0
    for s in characters:
        ci = gf.components.text(text=s)
        ci.name = s
        char = c << ci.flatten()  # type: ignore
        char.dx = x
        x += dx  # type: ignore

    return c


if __name__ == "__main__":
    c = alphabet()
    c.write_gds("alphabet.gds")
    c.show()
