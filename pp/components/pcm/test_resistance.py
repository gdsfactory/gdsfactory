from typing import Tuple, Optional
import numpy as np

from phidl.geometry import offset

import pp
from pp import components as pc
from pp.component import Component


@pp.autoname
def test_resistance(
    pad_size: Tuple[float] = (50.0, 50.0),
    num_squares: int = 1000,
    width: float = 1.0,
    res_layer: Optional[Tuple[int, int]] = 0,
    pad_layer: Optional[Tuple[int, int]] = None,
    gnd_layer: Optional[Tuple[int, int]] = None,
) -> Component:

    """ meander to test resistance
    from phidl.geometry

    Args:
        pad_size: Size of the two matched impedance pads (microns)
        num_squares: Number of squares comprising the resonator wire
        width: The width of the squares (microns)
        res_layer:
        pad_layer:
        gnd_layer:

    .. plot::
      :include-source:

      import pp

      c = pp.c.test_resistance()
      pp.plotgds(c)

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
    T = pp.Component()
    Row = pc.rectangle(size=(length_row, width), layer=res_layer)
    Col = pc.rectangle(size=(width, width), layer=res_layer)

    T.add_ref(Row)
    col = T.add_ref(Col)
    col.move([length_row - width, -width])

    # Creating entire waveguide net
    N = pp.Component("net")
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
    P = pp.Component()
    Pad1 = pc.rectangle(size=(x, z), layer=pad_layer)
    Pad2 = pc.rectangle(size=(x + 5, z), layer=pad_layer)
    Gnd1 = offset(Pad1, distance=-5, layer=gnd_layer)
    Gnd2 = offset(Pad2, distance=-5, layer=gnd_layer)
    pad1 = P.add_ref(Pad1).movex(-x - width)
    pad2 = P.add_ref(Pad1).movex(length_row + width)
    P.add_ref(Gnd1).center = pad1.center
    gnd2 = P.add_ref(Gnd2)
    P.add_ref(N).y = pad1.y
    gnd2.center = pad2.center
    gnd2.movex(2.5)

    return P


if __name__ == "__main__":
    c = test_resistance()
    pp.show(c)
