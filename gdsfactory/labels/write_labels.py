"""Find GDS labels and write them to a CSV file."""

import csv
import pathlib
from pathlib import Path
from typing import Iterator, Tuple

from loguru import logger

import gdsfactory as gf
from gdsfactory import LAYER
from gdsfactory.routing.add_fiber_single import add_fiber_single
from gdsfactory.types import Optional, PathType


def find_labels(
    gdspath: PathType, layer_label: Tuple[int, int] = LAYER.LABEL, prefix: str = "opt_"
) -> Iterator[Tuple[str, float, float]]:
    """Return text label and locations iterator from a GDS file.

    Klayout does not support label rotations.

    Args:
        gdspath: for the gds.
        layer_label: for the labels.
        prefix: for the labels to select.

    Returns
        string: for the label.
        x: x position (um).
        y: y position (um).

    """
    import klayout.db as pya

    # Load the layout
    gdspath = str(gdspath)
    layout = pya.Layout()
    layout.read(gdspath)

    # Get the top cell and the units, and find out the index of the layer
    topcell = layout.top_cell()
    dbu = layout.dbu
    layer = pya.LayerInfo(layer_label[0], layer_label[1])
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


def write_labels_klayout(
    gdspath: PathType,
    layer_label: Tuple[int, int] = LAYER.TEXT,
    filepath: Optional[PathType] = None,
    prefix: str = "opt_",
) -> Path:
    """Load GDS and extracts labels in klayout text and coordinates.

    Returns CSV filepath.

    Args:
        gdspath: for the mask.
        layer_label: for labels to write.
        filepath: for CSV file. Defaults to gdspath with CSV suffix.
        prefix: for the labels to write.

    """
    labels = list(find_labels(gdspath, layer_label=layer_label, prefix=prefix))
    gdspath = pathlib.Path(gdspath)

    filepath = filepath or gdspath.with_suffix(".csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
    logger.info(f"Wrote {len(labels)} labels to CSV {filepath.absolute()}")
    return filepath


def write_labels_gdspy(
    gdspath: Path,
    prefix: str = "opt_",
    layer_label: Optional[Tuple[int, int]] = LAYER.TEXT,
    filepath: Optional[PathType] = None,
    debug: bool = False,
    set_transform: bool = True,
) -> Path:
    """Load GDS and extracts label text and coordinates.

    Returns CSV filepath. Text, x, y, rotation (degrees)

    Args:
        gdspath: for the mask.
        prefix: for the labels to write.
        layer_label: for labels to write.
        filepath: for CSV file. Defaults to gdspath with CSV suffix.
        debug: prints the label.
        set_transform: bool
            If True, labels will include the transformations from
            the references they are from.

    """
    gdspath = pathlib.Path(gdspath)
    filepath = filepath or gdspath.with_suffix(".csv")
    filepath = pathlib.Path(filepath)
    c = gf.import_gds(gdspath)

    labels = []

    for label in c.get_labels(set_transform=set_transform):
        if (
            layer_label
            and label.layer == layer_label[0]
            and label.texttype == layer_label[1]
            and label.text.startswith(prefix)
        ):
            labels += [(label.text, label.x, label.y, label.rotation)]
            if debug:
                print(label.text, label.x, label.y, label.rotation)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
    logger.info(f"Wrote {len(labels)} labels to {filepath.absolute()}")
    return filepath


def test_find_labels() -> None:
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
    # cc.show(show_ports=True)
    # gdspath = CONFIG["samples_path"] / "mask" / "build" / "mask" / "sample.gds"
    # write_labels(gdspath)
