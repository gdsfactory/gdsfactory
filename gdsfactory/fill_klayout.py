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
    fill_cell_name: str = "fill_cell",
    create_new_fill_cell: bool = False,
    include_old_shapes: bool = False,
    fill_layers: LayerSpecs = None,
    fill_size: Tuple[float, float] = (10, 10),
    fill_spacing: Tuple[float, float] = (20, 20),
    fill_name: str = "fill",
    gdspath_out: Optional[PathType] = None,
) -> None:
    """Write gds file with fill.

    Args:
        gdspath: GDS input.
        layer_to_fill:
        layer_to_fill_margin:
        layers_to_avoid:
        layers_to_avoid_margin:
        cell_name: Optional cell to fill. Defaults to top cell.
        fill_cell_name: Optional cell name to use as fill.
        create_new_fill_cell: creates new fill cell, otherwise uses fill_cell_name from gdspath.
        fill_layers:
        fill_size:
        fill_spacing:
        fill_name: name of the cell containing all fill cells.
        gdspath_out: Optional GDS output. Defaults to input.
    """

    lib = kf.kcell.KLib()
    lib.read(filename=str(gdspath))
    cell = lib[cell_name or 0]

    fill = kf.KCell(fill_name)

    if create_new_fill_cell:
        if lib.has_cell(fill_cell_name):
            raise ValueError(f"{fill_cell_name!r} already in {str(gdspath)!r}")
        fill_cell = kf.KCell(fill_cell_name)
        for layer in fill_layers:
            layer = gf.get_layer(layer)
            layer = kf.klib.layer(*layer)
            fill_cell << kf.pcells.waveguide.waveguide(
                width=fill_size[0], length=fill_size[1], layer=layer
            )
    else:
        fill_cell = lib[fill_cell_name]

    fill_cell_index = fill_cell.cell_index()  # fill cell index
    fill_cell_box = fill_cell.bbox().enlarged(
        fill_spacing[0] / 2 * 1e3, fill_spacing[1] / 2 * 1e3
    )
    fill_margin = kf.kdb.Point(0, 0)

    layer_to_fill = gf.get_layer(layer_to_fill)
    layer_to_fill = cell.klib.layer(*layer_to_fill)
    region = kdb.Region()
    region_ = kdb.Region()

    if layers_to_avoid:
        for layer in layers_to_avoid:
            layer = gf.get_layer(layer)
            layer = kf.klib.layer(*layer)
            region_.insert(
                cell.begin_shapes_rec(layer)
            ) if layer != layer_to_fill else None

    region.insert(cell.begin_shapes_rec(layer_to_fill))
    region_to_fill = region - region_

    fill.fill_region(
        region_to_fill,
        fill_cell_index,
        fill_cell_box,
        fill_margin,
    )

    if include_old_shapes:
        for layer in cell.klib.layer_infos():
            fill.shapes(fill.klib.layer(layer)).insert(
                cell.begin_shapes_rec(cell.klib.layer(layer))
            )

    gdspath_out = gdspath_out or gdspath
    fill.write(str(gdspath_out))


@gf.cell
def cell_with_pad():
    c = gf.Component()
    c << gf.components.mzi(decorator=gf.add_padding)
    pad = c << gf.components.pad(size=(2, 2))
    pad.movey(10)
    return c


if __name__ == "__main__":
    c = cell_with_pad()
    c.show()
    gdspath = c.write_gds("mzi_fill.gds")
    fill(
        gdspath,
        fill_layers=("WG",),
        layer_to_fill=gf.LAYER.PADDING,
        layers_to_avoid=(gf.LAYER.PADDING, gf.LAYER.WG),
        fill_cell_name="fill_cell",
        create_new_fill_cell=True,
        fill_spacing=(1, 1),
        fill_size=(1, 1),
    )
    # fill(
    #     gdspath,
    #     fill_layers=("WG",),
    #     layer_to_fill=gf.LAYER.PADDING,
    #     fill_cell_name="pad_size2__2",
    #     create_new_fill_cell=False,
    #     fill_spacing=(1, 1),
    #     fill_size=(1, 1),
    # )
    gf.show(gdspath)

    # # import gdsfactory.fill_processor as fill
    # import kfactory.utils.geo.fill as fill

    # c = kf.KCell("ToFill")
    # c.shapes(kf.klib.layer(1, 0)).insert(
    #     kf.kdb.DPolygon.ellipse(kf.kdb.DBox(5000, 3000), 512)
    # )
    # c.shapes(kf.klib.layer(10, 0)).insert(
    #     kf.kdb.DPolygon(
    #         [kf.kdb.DPoint(0, 0), kf.kdb.DPoint(5000, 0), kf.kdb.DPoint(5000, 3000)]
    #     )
    # )

    # fc = kf.KCell("fill")
    # fc.shapes(fc.klib.layer(2, 0)).insert(kf.kdb.DBox(20, 40))
    # fc.shapes(fc.klib.layer(3, 0)).insert(kf.kdb.DBox(30, 15))

    # # fill.fill_tiled(c, fc, [(kf.klib.layer(1,0), 0)], exclude_layers = [(kf.klib.layer(10,0), 100), (kf.klib.layer(2,0), 0), (kf.klib.layer(3,0),0)], x_space=5, y_space=5)
    # fill.fill_tiled(
    #     c,
    #     fc,
    #     [(kf.klib.layer(1, 0), 0)],
    #     exclude_layers=[
    #         (kf.klib.layer(10, 0), 100),
    #         (kf.klib.layer(2, 0), 0),
    #         (kf.klib.layer(3, 0), 0),
    #     ],
    #     x_space=5,
    #     y_space=5,
    # )

    # gdspath = "mzi_fill.gds"
    # c.write(gdspath)
    # gf.show(gdspath)
