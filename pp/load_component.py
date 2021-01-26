"""Read component GDS, JSON metadata and CSV ports."""
import csv
import json
from pathlib import Path

import pp
from pp.component import Component


def remove_gds_labels(component: Component, layer=pp.LAYER.LABEL_SETTINGS) -> None:
    """Returns same component without labels."""
    for c in list(component.get_dependencies(recursive=True)) + [component]:
        old_label = [
            label for label in c.labels if label.layer == pp.LAYER.LABEL_SETTINGS
        ]
        if len(old_label) > 0:
            for label in old_label:
                c.labels.remove(label)


def load_component(gdspath: Path) -> Component:
    """Returns Component  with ports (CSV) and metadata (JSON) info (if any)."""

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


if __name__ == "__main__":
    from pp.tests.test_load_component import test_load_component_gds

    test_load_component_gds()
