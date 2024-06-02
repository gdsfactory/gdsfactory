"""Find GDS labels and write them to a CSV file."""

from __future__ import annotations

import csv
import pathlib
from collections.abc import Iterator
from pathlib import Path

import klayout.db as pya

from gdsfactory import logger
from gdsfactory.typings import PathType


def find_labels(
    gdspath: PathType, layer_label: tuple[int, int], prefix: str = ""
) -> Iterator[tuple[str, float, float, float]]:
    """Return text label and locations iterator from a GDS file.

    Klayout does not support label rotations.

    Args:
        gdspath: for the gds.
        layer_label: for the labels.
        prefix: for the labels to select.

    Returns:
        string: for the label.
        x: x position (um).
        y: y position (um).
        angle: in degrees.
    """
    # Load the layout
    gdspath = str(gdspath)
    layout = pya.Layout()
    layout.read(gdspath)

    # Get the top cell and the units, and find out the index of the layer
    topcell = layout.top_cell()

    # Extract locations
    iterator = topcell.begin_shapes_rec(layout.layer(*layer_label))

    while not (iterator.at_end()):
        shape, trans = iterator.shape(), iterator.dtrans()
        iterator.next()
        if shape.is_text():
            text = shape.dtext
            if text.string.startswith(prefix):
                transformed = text.transformed(trans)
                yield text.string, transformed.x, transformed.y, trans.angle


def write_labels(
    gdspath: PathType,
    layer_label: tuple[int, int] = (66, 0),
    filepath: PathType | None = None,
    prefix: str = "opt_",
) -> Path:
    """Load GDS and extracts labels in KLayout text and coordinates.

    Returns CSV filepath with each row:
    - Text
    - x
    - y
    - rotation (degrees)

    Args:
        gdspath: for the mask.
        layer_label: for labels to write.
        filepath: for CSV file. Defaults to gdspath with CSV suffix.
        prefix: filter labels with this prefix.

    """
    labels = list(find_labels(gdspath, layer_label=layer_label, prefix=prefix))
    gdspath = pathlib.Path(gdspath)
    filepath = Path(filepath or gdspath.with_suffix(".csv"))

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
    logger.info(f"Wrote {len(labels)} labels to CSV {filepath.absolute()}")
    return filepath
