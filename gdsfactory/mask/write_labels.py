"""
find GDS labels and write labels into a CSV file
"""

import csv
import pathlib
from pathlib import Path
from typing import Iterator, Tuple

import klayout.db as pya
from loguru import logger

from gdsfactory import LAYER
from gdsfactory.routing.add_fiber_single import add_fiber_single


def find_labels(
    gdspath: Path, label_layer: Tuple[int, int] = LAYER.LABEL, prefix: str = "opt_"
) -> Iterator[Tuple[str, float, float]]:
    """finds labels and locations from a GDS file"""
    # Load the layout
    gdspath = str(gdspath)
    layout = pya.Layout()
    layout.read(gdspath)

    # Get the top cell and the units, and find out the index of the layer
    topcell = layout.top_cell()
    dbu = layout.dbu
    layer = pya.LayerInfo(label_layer[0], label_layer[1])
    layer_index = layout.layer(layer)

    # Extract locations
    iterator = topcell.begin_shapes_rec(layer_index)

    while not (iterator.at_end()):
        shape, trans = iterator.shape(), iterator.trans()
        iterator.next()
        if shape.is_text():
            text = shape.text
            if text.string.startswith(prefix):
                transformed = text.transformed(trans)
                yield text.string, transformed.x * dbu, transformed.y * dbu


def write_labels(
    gdspath: Path,
    label_layer: Tuple[int, int] = LAYER.LABEL,
    csv_filename: None = None,
    prefix: str = "opt_",
) -> None:
    """Load  GDS mask and extracts the labels and coordinates from a GDS file"""
    labels = list(find_labels(gdspath, label_layer=label_layer, prefix=prefix))

    # Save the coordinates somewhere sensible
    if csv_filename is None:
        gdspath = pathlib.Path(gdspath)
        csv_filename = gdspath.with_suffix(".csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
    logger.info(f"Wrote {csv_filename}")


def test_find_labels():
    import gdsfactory as gf

    c = gf.components.straight(length=124)
    cc = add_fiber_single(component=c)
    gdspath = cc.write_gds()
    assert len(list(find_labels(gdspath))) == 4


if __name__ == "__main__":
    test_find_labels()

    # import gdsfactory as gf

    # c = gf.components.straight()
    # cc = add_fiber_single(component=c)
    # gdspath = cc.write_gds()
    # print(len(list(find_labels(gdspath))))
    # cc.show()

    # gdspath = CONFIG["samples_path"] / "mask" / "build" / "mask" / "sample.gds"
    # write_labels(gdspath)
