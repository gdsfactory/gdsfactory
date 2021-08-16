from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.tech import LAYER


@gf.cell
def pads_shorted(
    width: int = 100,
    n_pads: int = 8,
    pad_spacing: int = 150,
    layer: Tuple[int, int] = LAYER.M1,
) -> Component:
    c = Component(name="shorted_pads")
    pad = rectangle(size=(width, width), layer=layer, centered=True)
    for i in range(n_pads):
        pad_ref = c.add_ref(pad)
        pad_ref.movex(i * pad_spacing - n_pads / 2 * pad_spacing + pad_spacing / 2)

    short = rectangle(size=(pad_spacing * (n_pads - 1), 10), layer=layer, centered=True)
    c.add_ref(short)
    return c


if __name__ == "__main__":

    c = pads_shorted()
    c.show()
