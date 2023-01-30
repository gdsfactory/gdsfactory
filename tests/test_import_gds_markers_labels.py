from __future__ import annotations

import pandas as pd
import yaml

import gdsfactory as gf


def test_import_ports_markers_labels(data_regression) -> gf.Component:
    """Make sure you can import the ports"""
    c = gf.components.straight(
        decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    )
    c1 = gf.labels.add_label_yaml(
        gf.functions.rotate(gf.routing.add_fiber_single(c), angle=90)
    )
    gdspath = c1.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefix="component_name"
    )
    labels = pd.read_csv(csvpath)

    settings = yaml.safe_load(labels.columns[0])
    data_regression.check(settings)


if __name__ == "__main__":
    c = gf.components.straight(
        decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    )
    c1 = gf.labels.add_label_yaml(
        gf.functions.rotate(gf.routing.add_fiber_single(c), angle=90)
    )
    gdspath = c1.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefix="component_name"
    )
    labels = pd.read_csv(csvpath)
    settings = yaml.safe_load(labels.columns[0])
    c1.show()
