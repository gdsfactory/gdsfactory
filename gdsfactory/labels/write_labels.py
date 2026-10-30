"""Find GDS labels and write them to a CSV file."""

from __future__ import annotations

import csv
import pathlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import klayout.db as pya

import gdsfactory as gf
from gdsfactory import logger
from gdsfactory.typings import LayerSpec, PathType


def find_labels(
    gdspath: PathType,
    layer_label: LayerSpec = (66, 0),
    prefixes: Iterable[str] = ("opt-", "elec-"),
) -> Iterator[tuple[str, float, float, float]]:
    """Return text label and locations iterator from a GDS file.

    Klayout does not support label rotations.

    Args:
        gdspath: for the gds.
        layer_label: for the labels.
        prefixes: prefixes to extract labels.

    Returns:
        string: for the label.
        x: x position (um).
        y: y position (um).
        angle: in degrees.
    """
    gdspath = str(gdspath)
    layout = pya.Layout(False)
    read_layer = layer_label
    if type(layer_label) is str:
        try:
            read_layer = gf.get_layer_tuple(layer_label)
        except Exception:
            # Preserve read errors before reporting an invalid layer.
            pass
    # Select known numeric layers before loading geometry.
    if (
        type(read_layer) is tuple
        and len(read_layer) == 2
        and all(type(value) is int and 0 <= value <= 65535 for value in read_layer)
    ):
        options = pya.LoadLayoutOptions()
        options.properties_enabled = False
        layer_map = pya.LayerMap()
        layer_map.map(pya.LayerInfo(*read_layer), 0)
        options.set_layer_map(layer_map, False)
        layout.read(gdspath, options)
    else:
        layout.read(gdspath)
    topcell = layout.top_cell()
    layer_label = gf.get_layer_tuple(layer_label)
    iterator = topcell.begin_shapes_rec(layout.layer(*layer_label))
    iterator.shape_flags = pya.Shapes.STexts

    # each() advances the same recursive iterator in place.
    shape = iterator.shape
    dtrans = iterator.dtrans
    for _ in iterator.each():
        text = shape().dtext
        string = text.string
        result = None
        for prefix in prefixes:
            if string.startswith(prefix):
                if result is None:
                    trans = dtrans()
                    transformed = text.transformed(trans)
                    result = string, transformed.x, transformed.y, trans.angle
                yield result


def write_labels(
    gdspath: PathType,
    layer_label: LayerSpec = (66, 0),
    filepath: PathType | None = None,
    prefixes: Iterable[str] = ("opt-", "elec-"),
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
        prefixes: prefixes to extract labels.

    """
    labels = list(find_labels(gdspath, layer_label=layer_label, prefixes=prefixes))
    gdspath = pathlib.Path(gdspath)
    filepath = Path(filepath or gdspath.with_suffix(".csv"))

    columns = ["text", "x", "y", "angle"]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(labels)
    logger.info(f"Wrote {len(labels)} labels to CSV {filepath.absolute()!r}")
    return filepath
