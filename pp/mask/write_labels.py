"""
find GDS labels and write labels into a CSV file
"""

import csv
import pathlib

import klayout.db as pya

from pp import LAYER


def find_labels(gdspath, label_layer=LAYER.LABEL, prefix="opt_"):
    """ finds labels and locations from a GDS file """
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


def write_labels(gdspath, label_layer=LAYER.LABEL, csv_filename=None, prefix="opt_"):
    """Load  GDS mask and extracts the labels and coordinates from a GDS file"""
    labels = list(find_labels(gdspath, label_layer=label_layer, prefix=prefix))

    # Save the coordinates somewhere sensible
    if csv_filename is None:
        gdspath = pathlib.Path(gdspath)
        csv_filename = gdspath.with_suffix(".csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
    print(f"Wrote {csv_filename}")


if __name__ == "__main__":
    from pp.config import CONFIG

    gdspath = CONFIG["samples_path"] / "mask" / "build" / "mask" / "sample.gds"
    write_labels(gdspath)
