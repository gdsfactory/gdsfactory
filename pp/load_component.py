""" load component GDS, JSON metadata and CSV ports
"""
import csv
import json
import os
from pathlib import Path

from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

import pp
from pp import CONFIG
from pp.component import Component


def get_component_path(name, dirpath=CONFIG["gdslib"]):
    return dirpath / f"{name}.gds"


def load_component_path(name, dirpath=CONFIG["gdslib"]):
    """load component GDS from a library
    returns a gdspath
    """
    gdspath = dirpath / f"{name}.gds"

    if not os.path.isfile(gdspath):
        raise ValueError(f"cannot load `{gdspath}`")

    return gdspath


def remove_gds_labels(component: Component, layer=pp.LAYER.LABEL_SETTINGS) -> None:
    """Returns same component without labels"""
    for component in list(component.get_dependencies(recursive=True)) + [component]:
        old_label = [
            label
            for label in component.labels
            if label.layer == pp.LAYER.LABEL_SETTINGS
        ]
        if len(old_label) > 0:
            for label in old_label:
                component.labels.remove(label)


def load_component(gdspath: Path) -> Component:
    """Returns Component from gdspath, with ports (CSV) and metadata (JSON) info (if any)"""

    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")

    ports_filepath = gdspath.with_suffix(".ports")
    metadata_filepath = gdspath.with_suffix(".json")

    c = pp.import_gds(gdspath)

    if ports_filepath.exists():
        with open(str(ports_filepath), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for r in reader:
                layer_type = int(r[5].strip().strip("("))
                data_type = int(r[6].strip().strip(")"))
                c.add_port(
                    name=r[0],
                    midpoint=[float(r[1]), float(r[2])],
                    orientation=int(r[3]),
                    width=float(r[4]),
                    layer=(layer_type, data_type),
                )

    if metadata_filepath.exists():
        with open(metadata_filepath) as f:
            data = json.load(f)
        cell_settings = data["cells"][c.name]
        c.settings.update(cell_settings)
    return c


def test_load_component_gds() -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    assert c.hash_geometry() == "acbc8481cf28bc9930ecbd373cafcd17b39c5c27"


def test_load_component_ports(num_regression: NumericRegressionFixture) -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    num_regression.check(c.get_ports_array())


def test_load_component_settings(data_regression: DataRegressionFixture) -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    data_regression.check(c.get_settings())


if __name__ == "__main__":
    test_load_component_gds()
