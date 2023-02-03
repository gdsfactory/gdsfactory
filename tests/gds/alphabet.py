from __future__ import annotations

import gdsfactory as gf

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@gf.cell
def alphabet(dx=10):
    c = gf.Component()
    x = 0
    for s in characters:
        ci = gf.components.text(text=s)
        ci.name = s
        char = c << ci.flatten()
        char.x = x
        x += dx

    return c


if __name__ == "__main__":
    c = alphabet()
    c.write_gds("alphabet.gds")
    c.show()
