from pydantic import validate_arguments

import pp
from pp.cell import cell

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@cell
@validate_arguments
def alphabet(dx=10):
    c = pp.Component()
    x = 0
    for s in characters:
        ci = pp.components.text(text=s)
        ci.name = s
        char = c << ci.flatten()
        char.x = x
        x += dx

    return c


if __name__ == "__main__":
    c = alphabet()
    c.write_gds("alphabet.gds")
    c.show()
