from functools import partial
from typing import Tuple

import numpy as np

from gdsfactory import components as pc
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.tech import LAYER
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.types import ComponentFactory


pad80 = partial(pad_function, size=(80, 80))


@cell
def resistance_meander(
    num_squares: int = 1000,
    width: float = 1.0,
    layer: Tuple[int, int] = LAYER.M3,
    pad: ComponentFactory = pad80,
) -> Component:
    """meander to test resistance
    from phidl.geometry

    FIXME, add pad_pitch

    Args:
        pad_size: Size of the two matched impedance pads (microns)
        num_squares: Number of squares comprising the resonator wire
        width: The width of the squares (microns)
        layer:
        pad: function for pad

    """
    pad = pad()

    pad_size = pad.info.size
    x = pad_size[0]
    z = pad_size[1]

    # Checking validity of input
    if x <= 0 or z <= 0:
        raise ValueError("Pad must have positive, real dimensions")
    elif width > z:
        raise ValueError("Width of cell cannot be greater than height of pad")
    elif num_squares <= 0:
        raise ValueError("Number of squares must be a positive real number")
    elif width <= 0:
        raise ValueError("Width of cell must be a positive real number")

    # Performing preliminary calculations
    num_rows = int(np.floor(z / (2 * width)))
    if num_rows % 2 == 0:
        num_rows -= 1
    num_columns = num_rows - 1
    squares_in_row = (num_squares - num_columns - 2) / num_rows

    # Compensating for weird edge cases
    if squares_in_row < 1:
        num_rows = round(num_rows / 2) - 2
        squares_in_row = 1
    if width * 2 > z:
        num_rows = 1
        squares_in_row = num_squares - 2

    length_row = squares_in_row * width

    # Creating row/column corner combination structure
    T = Component()
    Row = pc.rectangle(size=(length_row, width), layer=layer, port_type=None)
    Col = pc.rectangle(size=(width, width), layer=layer, port_type=None)

    T.add_ref(Row)
    col = T.add_ref(Col)
    col.move([length_row - width, -width])

    # Creating entire straight net
    N = Component("net")
    n = 1
    for i in range(num_rows):
        if i != num_rows - 1:
            d = N.add_ref(T)
        else:
            d = N.add_ref(Row)
        if n % 2 == 0:
            d.reflect(p1=(d.x, d.ymax), p2=(d.x, d.ymin))
        d.movey(-(n - 1) * T.ysize)
        n += 1
    d = N.add_ref(Col).movex(-width)
    d = N.add_ref(Col).move([length_row, -(n - 2) * T.ysize])

    # Creating pads
    P = Component()
    pad1 = P.add_ref(pad)
    pad1.movex(-x - width)
    pad2 = P.add_ref(pad)
    pad2.movex(length_row + width)

    pad1.xmax = 0
    pad2.xmin = z

    net = P.add_ref(N)
    net.y = pad1.y
    P.absorb(net)
    P.absorb(pad1)
    P.absorb(pad2)
    return P


if __name__ == "__main__":
    c = resistance_meander()
    c.show()
