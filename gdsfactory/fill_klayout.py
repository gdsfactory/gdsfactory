"""Dummy fill to keep density constant using klayout."""
from __future__ import annotations

from typing import Optional, Tuple


import gdsfactory as gf
from gdsfactory.typings import PathType, LayerSpecs, Layer

import kfactory as kf
import klayout.db as kdb


def fill(
    gdspath,
    layer_to_fill: Layer,
    layer_to_fill_margin: float = 0,
    layers_to_avoid: LayerSpecs = None,
    layers_to_avoid_margin: float = 0,
    cell_name: Optional[str] = None,
    fill_cell_name: Optional[str] = None,
    fill_layers: LayerSpecs = None,
    fill_size: Tuple[float, float] = (10, 10),
    fill_spacing: Tuple[float, float] = (20, 20),
    fill_name: str = "fill",
    gdspath_out: Optional[PathType] = None,
) -> None:
    """Write gds file with fill.

    Args:
        gdspath: GDS input.
        gdspath_out: Optional GDS output. Defaults to input.
        cell_name: Optional cell to fill. Defaults to top cell.
        fill_cell_name: Optional cell name to use as fill.
        fill_layers:
        fill_size:
        fill_spacing:
        fill_name: name of the cell containing all fill cells.
    """

    lib = kf.kcell.KLib()
    lib.read(filename=str(gdspath))
    cell = lib[cell_name or 0]

    if fill_cell_name:
        fill_cell = lib[fill_cell_name]

    else:
        fill_cell = kf.KCell("fill_cell")
        for layer in fill_layers:
            layer = gf.get_layer(layer)
            layer = kf.klib.layer(*layer)
            fill_cell << kf.pcells.waveguide.waveguide(
                width=fill_size[0], length=fill_size[1], layer=layer
            )

    fill_cell_index = fill_cell.cell_index()  # fill cell index
    fill_cell_box = fill_cell.bbox().enlarged(
        fill_spacing[0] / 2 * 1e3, fill_spacing[1] / 2 * 1e3
    )
    fill_margin = kf.kdb.Point(0, 0)

    region = kdb.Region()
    layer = gf.get_layer(layer_to_fill)
    layer = kf.klib.layer(*layer)
    region.insert(cell.begin_shapes_rec(layer))
    region.merge()
    # region.size(layer_to_fill_margin)

    fill = kf.KCell(fill_name)
    fill.fill_region(
        region, fill_cell_index, fill_cell_box, None, region, fill_margin, None
    )
    gdspath_out = gdspath_out or gdspath
    fill.write(str(gdspath_out))


if __name__ == "__main__":
    c = gf.components.mzi(decorator=gf.add_padding)
    gdspath = c.write_gds("mzi_fill.gds")
    fill(
        gdspath,
        fill_layers=("WG",),
        layer_to_fill=gf.LAYER.PADDING,
    )
    gf.show(gdspath)
