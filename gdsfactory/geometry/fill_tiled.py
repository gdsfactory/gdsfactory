from typing import Iterable, Optional

try:
    import kfactory as kf
    from kfactory import KCell, kcl, LayerEnum, kdb
    from kfactory.conf import logger
except ImportError as e:
    print(
        "You can install `pip install gdsfactory[kfactory]` for using maskprep. "
        "And make sure you use python >= 3.10"
    )
    raise e


class FillOperator(kdb.TileOutputReceiver):
    def __init__(
        self,
        kcl: kcl,
        top_cell: KCell,
        fill_cell_index: int,
        fc_bbox: kdb.Box,
        row_step: kdb.Vector,
        column_step: kdb.Vector,
        fill_margin: kdb.Vector = kdb.Vector(0, 0),
        remaining_polygons: Optional[kdb.Region] = None,
    ) -> None:
        self.kcl = kcl
        self.top_cell = top_cell
        self.fill_cell_index = fill_cell_index
        self.fc_bbox = fc_bbox
        self.row_step = row_step
        self.column_step = column_step
        self.fill_margin = fill_margin
        self.remaining_polygons = remaining_polygons
        self.glue_box = self.top_cell.bbox()

    def put(
        self,
        ix: int,
        iy: int,
        tile: kdb.Box,
        region: kdb.Region,
        dbu: float,
        clip: bool,
    ) -> None:
        while not region.is_empty():
            self.top_cell.fill_region(
                region,
                self.fill_cell_index,
                self.fc_bbox,
                self.row_step,
                self.column_step,
                tile.p1,
                region,
                self.fill_margin,
                None,
                self.glue_box,
            )


def fill_tiled(
    c: KCell,
    fill_cell: KCell,
    fill_layers: Iterable[tuple[LayerEnum | int, int]] = [],
    fill_regions: Iterable[tuple[kdb.Region, int]] = [],
    exclude_layers: Iterable[tuple[LayerEnum | int, int]] = [],
    exclude_regions: Iterable[tuple[kdb.Region, int]] = [],
    n_threads: int = 4,
    tile_size: Optional[tuple[float, float]] = None,
    x_space: float = 0,
    y_space: float = 0,
) -> None:
    tp = kdb.TilingProcessor()
    tp.frame = c.bbox().to_dtype(c.kcl.dbu)  # type: ignore
    tp.dbu = c.kcl.dbu
    tp.threads = n_threads

    if tile_size is None:
        tile_size = (
            100 * (fill_cell.dbbox().width() + x_space),
            100 * (fill_cell.dbbox().height() + y_space),
        )
    tp.tile_size(*tile_size)

    layer_names: list[str] = []
    for layer, _ in fill_layers:
        layer_name = f"layer{layer}"
        tp.input(layer_name, c.kcl, c.cell_index(), c.kcl.get_info(layer))
        layer_names.append(layer_name)

    region_names: list[str] = []
    for i, (r, _) in enumerate(fill_regions):
        region_name = f"region{i}"
        tp.input(region_name, r)
        region_names.append(region_name)

    exlayer_names: list[str] = []
    for layer, _ in exclude_layers:
        layer_name = f"layer{layer}"
        tp.input(layer_name, c.kcl, c.cell_index(), c.kcl.get_info(layer))
        exlayer_names.append(layer_name)

    exregion_names: list[str] = []
    for i, (r, _) in enumerate(exclude_regions):
        region_name = f"region{i}"
        tp.input(region_name, r)
        exregion_names.append(region_name)

    tp.output(
        "to_fill",
        FillOperator(
            c.kcl,
            c,
            fill_cell.cell_index(),
            fc_bbox=fill_cell.bbox(),
            row_step=kdb.Vector(fill_cell.bbox().width() + int(x_space / c.kcl.dbu), 0),
            column_step=kdb.Vector(
                0, fill_cell.bbox().height() + int(y_space / c.kcl.dbu)
            ),
        ),
    )

    if layer_names or region_names:
        exlayers = " + ".join(
            [
                layer_name + f".sized({int(size / c.kcl.dbu)})" if size else layer_name
                for layer_name, (_, size) in zip(exlayer_names, exclude_layers)
            ]
        )
        exregions = " + ".join(
            [
                region_name + f".sized({int(size / c.kcl.dbu)})"
                if size
                else region_name
                for region_name, (_, size) in zip(exregion_names, exclude_regions)
            ]
        )
        layers = " + ".join(
            [
                layer_name + f".sized({int(size / c.kcl.dbu)})" if size else layer_name
                for layer_name, (_, size) in zip(layer_names, fill_layers)
            ]
        )
        regions = " + ".join(
            [
                region_name + f".sized({int(size / c.kcl.dbu)})"
                if size
                else region_name
                for region_name, (_, size) in zip(region_names, fill_regions)
            ]
        )

        if exlayer_names or exregion_names:
            queue_str = (
                "var fill= "
                + (
                    " + ".join([layers, regions])
                    if regions and layers
                    else regions + layers
                )
                + "; var exclude = "
                + (
                    " + ".join([exlayers, exregions])
                    if exregions and exlayers
                    else exregions + exlayers
                )
                + "; var fill_region = _tile & _frame & fill - exclude; _output(to_fill, fill_region)"
            )
        else:
            queue_str = (
                "var fill= "
                + (
                    " + ".join([layers, regions])
                    if regions and layers
                    else regions + layers
                )
                + "; var fill_region = _tile & _frame & fill; _output(to_fill, fill_region)"
            )
        tp.queue(queue_str)
        c.kcl.start_changes()
        try:
            logger.info("filling {} with {}", c.name, fill_cell.name)
            logger.debug("fill string: '{}'", queue_str)
            tp.execute(f"Fill {c.name}")
            logger.info("done with filling {}", c.name)
        finally:
            c.kcl.end_changes()


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.geometry.fill_tiled as fill

    c = kf.KCell("ToFill")
    c.shapes(kf.kcl.layer(1, 0)).insert(
        kf.kdb.DPolygon.ellipse(kf.kdb.DBox(5000, 3000), 512)
    )
    c.shapes(kf.kcl.layer(10, 0)).insert(
        kf.kdb.DPolygon(
            [kf.kdb.DPoint(0, 0), kf.kdb.DPoint(5000, 0), kf.kdb.DPoint(5000, 3000)]
        )
    )

    fc = kf.KCell("fill")
    fc.shapes(fc.kcl.layer(2, 0)).insert(kf.kdb.DBox(20, 40))
    fc.shapes(fc.kcl.layer(3, 0)).insert(kf.kdb.DBox(30, 15))

    # fill.fill_tiled(c, fc, [(kf.kcl.layer(1,0), 0)], exclude_layers = [(kf.kcl.layer(10,0), 100), (kf.kcl.layer(2,0), 0), (kf.kcl.layer(3,0),0)], x_space=5, y_space=5)
    fill.fill_tiled(
        c,
        fc,
        [(kf.kcl.layer(1, 0), 0)],
        exclude_layers=[
            (kf.kcl.layer(10, 0), 100),
            (kf.kcl.layer(2, 0), 0),
            (kf.kcl.layer(3, 0), 0),
        ],
        x_space=5,
        y_space=5,
    )

    gdspath = "mzi_fill.gds"
    c.write(gdspath)
    gf.show(gdspath)
