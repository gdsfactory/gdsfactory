"""
find GDS labels and write labels into a CSV file
"""

import os
import csv
import klayout.db as pya


def find_labels(gdspath, label_layer=201, label_purpose=0):
    """ finds labels and locations from a GDS file """
    # Load the layout
    gdspath = str(gdspath)
    layout = pya.Layout()
    layout.read(gdspath)

    # Get the top cell and the units, and find out the index of the layer
    topcell = layout.top_cell()
    dbu = layout.dbu
    layer = pya.LayerInfo(label_layer, label_purpose)
    layer_index = layout.layer(layer)

    # Extract locations
    iterator = topcell.begin_shapes_rec(layer_index)

    while not (iterator.at_end()):
        shape, trans = iterator.shape(), iterator.trans()
        iterator.next()
        if shape.is_text():
            text = shape.text
            transformed = text.transformed(trans)
            yield text.string, transformed.x * dbu, transformed.y * dbu


def _write_csv(labels, filename):
    """ Writes labels to disk in JSON format """
    data = labels
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print(("Wrote {}".format(os.path.abspath(filename))))


def write_labels(gdspath, label_layer=201, label_purpose=0, csv_filename=None):
    """Load  GDS mask and extracts the labels and coordinates from a GDS file"""
    labels = list(
        find_labels(
            gdspath, label_layer=int(label_layer), label_purpose=int(label_purpose)
        )
    )

    # Save the coordinates somewhere sensible
    if csv_filename is None:
        csv_filename = gdspath.with_suffix(".csv")
    _write_csv(labels, csv_filename)


if __name__ == "__main__":
    from pp.config import CONFIG, load_config
    config_path = CONFIG["samples_path"] / "mask" / "config.yml"
    config = load_config(config_path)
    gdspath = str(config['mask']['gds'])
    write_labels(gdspath)
