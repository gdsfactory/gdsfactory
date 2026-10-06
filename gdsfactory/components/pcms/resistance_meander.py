from __future__ import annotations

__all__ = ["resistance_meander"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell
def _resistance_meander_row(
    length_row: float, width: float, res_layer: LayerSpec
) -> Component:
    """Returns a meander row with the corner square that joins it to the next row.

    Args:
        length_row: length of the row (microns).
        width: width of the squares (microns).
        res_layer: resistance layer.
    """
    c = Component()
    c.add_ref(gf.c.rectangle(size=(length_row, width), layer=res_layer))
    col = c.add_ref(gf.c.rectangle(size=(width, width), layer=res_layer))
    col.move((length_row - width, -width))
    return c


@gf.cell
def _resistance_meander_net(
    num_rows: int, length_row: float, width: float, res_layer: LayerSpec
) -> Component:
    """Returns the meander wire, without the pads.

    Args:
        num_rows: number of rows in the meander.
        length_row: length of each row (microns).
        width: width of the squares (microns).
        res_layer: resistance layer.
    """
    row_and_corner = _resistance_meander_row(
        length_row=length_row, width=width, res_layer=res_layer
    )
    row = gf.c.rectangle(size=(length_row, width), layer=res_layer)
    col = gf.c.rectangle(size=(width, width), layer=res_layer)

    c = Component()
    n = 1
    for i in range(num_rows):
        d = c.add_ref(row_and_corner) if i != num_rows - 1 else c.add_ref(row)
        if n % 2 == 0:
            d.dmirror_x(d.x)
        d.movey(-(n - 1) * row_and_corner.ysize)
        n += 1

    start = c.add_ref(col)
    start.movex(-width)

    end = c.add_ref(col)
    end.movey(-(n - 2) * row_and_corner.ysize)
    end.movex(length_row)
    return c


@gf.cell_with_module_name(tags=["pcms"])
def resistance_meander(
    pad_size: Size = (50.0, 50.0),
    num_squares: int = 1000,
    width: float = 1.0,
    res_layer: LayerSpec = "MTOP",
    pad_layer: LayerSpec = "MTOP",
) -> Component:
    """Return meander to test resistance.

    based on phidl.geometry

    Args:
        pad_size: Size of the two matched impedance pads (microns).
        num_squares: Number of squares comprising the resonator wire.
        width: The width of the squares (microns).
        res_layer: resistance layer.
        pad_layer: pad layer.
    """
    x = pad_size[0]
    z = pad_size[1]

    # Checking validity of input
    if x <= 0 or z <= 0:
        raise ValueError("Pad must have positive, real dimensions")
    if width > z:
        raise ValueError("Width of cell cannot be greater than height of pad")
    if num_squares <= 0:
        raise ValueError("Number of squares must be a positive real number")
    if width <= 0:
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

    net = _resistance_meander_net(
        num_rows=num_rows, length_row=length_row, width=width, res_layer=res_layer
    )

    # Creating pads
    c = Component()
    pad = gf.c.rectangle(size=(x, z), layer=pad_layer)
    pad1 = c.add_ref(pad)
    pad1.movex(-x - width)
    pad2 = c.add_ref(pad)
    pad2.movex(length_row + width)
    net_ref = c.add_ref(net)
    net_ref.ymin = pad1.ymin
    c.flatten()
    return c
