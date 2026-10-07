from __future__ import annotations

__all__ = ["resistance_meander", "resistance_meander_net", "resistance_meander_row"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell
def resistance_meander_row(
    length_row: float, width: float, res_layer: LayerSpec
) -> Component:
    """Returns a meander row with the corner square that joins it to the next row.

    Args:
        length_row: length of the row (microns).
        width: width of the squares (microns).
        res_layer: resistance layer.
    """
    return cf.resistance_meander_row(
        length_row=length_row,
        width=width,
        res_layer=res_layer,
    )


@gf.cell
def resistance_meander_net(
    num_rows: int, length_row: float, width: float, res_layer: LayerSpec
) -> Component:
    """Returns the meander wire, without the pads.

    Args:
        num_rows: number of rows in the meander.
        length_row: length of each row (microns).
        width: width of the squares (microns).
        res_layer: resistance layer.
    """
    return cf.resistance_meander_net(
        num_rows=num_rows,
        length_row=length_row,
        width=width,
        res_layer=res_layer,
    )


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
    return cf.resistance_meander(
        pad_size=pad_size,
        num_squares=num_squares,
        width=width,
        res_layer=res_layer,
        pad_layer=pad_layer,
    )
