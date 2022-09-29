from typing import Tuple

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.geometry.offset import offset
from gdsfactory.types import LayerSpec


@cell
def resistance_meander(
    pad_size: Tuple[float, float] = (50.0, 50.0),
    num_squares: int = 1000,
    width: float = 1.0,
    res_layer: LayerSpec = "M3",
    pad_layer: LayerSpec = "M3",
    gnd_layer: LayerSpec = "M3",
) -> Component:
    """Return meander to test resistance.

    based on phidl.geometry

    Args:
        pad_size: Size of the two matched impedance pads (microns).
        num_squares: Number of squares comprising the resonator wire.
        width: The width of the squares (microns).
        res_layer: resistance layer.
        pad_layer: pad layer.
        gnd_layer: ground layer.
    """
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
    Row = rectangle(size=(length_row, width), layer=res_layer)
    Col = rectangle(size=(width, width), layer=res_layer)

    T.add_ref(Row)
    col = T.add_ref(Col)
    col.move([length_row - width, -width])

    # Creating entire straight net
    N = Component("net")
    n = 1
    for i in range(num_rows):
        d = N.add_ref(T) if i != num_rows - 1 else N.add_ref(Row)
        if n % 2 == 0:
            d.reflect(p1=(d.x, d.ymax), p2=(d.x, d.ymin))
        d.movey(-(n - 1) * T.ysize)
        n += 1
    d = N.add_ref(Col).movex(-width)
    d = N.add_ref(Col).move([length_row, -(n - 2) * T.ysize])

    # Creating pads
    P = Component()
    pad1 = rectangle(size=(x, z), layer=pad_layer)
    gnd1 = gnd2 = offset(pad1, distance=-5, layer=gnd_layer)
    pad1_ref = P.add_ref(pad1)
    pad1_ref.movex(-x - width)
    pad2_ref = P.add_ref(pad1)
    pad2_ref.movex(length_row + width)
    gnd1_ref = P.add_ref(gnd1)
    gnd1_ref.center = pad1_ref.center
    gnd2_ref = P.add_ref(gnd2)
    net = P.add_ref(N)
    net.y = pad1_ref.y
    gnd2_ref.center = pad2_ref.center
    gnd2_ref.movex(2.5)
    P.absorb(net)
    P.absorb(gnd1_ref)
    P.absorb(gnd2_ref)
    P.absorb(pad1_ref)
    P.absorb(pad2_ref)
    return P


if __name__ == "__main__":
    c = resistance_meander(res_layer="M1", width=0.5)
    c.show(show_ports=True)
