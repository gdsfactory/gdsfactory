"""Read component GDS, JSON metadata and CSV ports."""
import csv
import json
from pathlib import Path
from typing import Union

from gdsfactory.component import Component


def read_ports(component: Component, gdspath: Union[str, Path]) -> None:
    """Adds Component with ports (CSV) and metadata (JSON) info (if any)."""

    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")

    ports_filepath = gdspath.with_suffix(".ports")
    metadata_filepath = gdspath.with_suffix(".json")

    if ports_filepath.exists():
        with open(str(ports_filepath), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for r in reader:
                layer_type = int(r[5].strip().strip("("))
                data_type = int(r[6].strip().strip(")"))
                component.add_port(
                    name=r[0],
                    midpoint=[float(r[1]), float(r[2])],
                    orientation=int(r[3]),
                    width=float(r[4]),
                    layer=(layer_type, data_type),
                )

    if metadata_filepath.exists():
        with open(metadata_filepath) as f:
            data = json.load(f)
        cell_settings = data["cells"][component.name]
        component.settings.update(cell_settings)


if __name__ == "__main__":
    from gdsfactory.tests.test_load_component import test_load_component_gds

    test_load_component_gds()
