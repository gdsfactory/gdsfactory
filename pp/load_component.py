"""
load component gds, and ports
"""
import os
import pathlib
import json
import csv

from pp import CONFIG
import pp
from pathlib import PosixPath
from pp.component import Component


def get_component_path(component_name, component_path=CONFIG["gdslib"]):
    return component_path / (component_name + ".gds")


def load_component_path(component_name, component_path=CONFIG["gdslib"]):
    """ load component GDS from a library
    returns a gdspath
    """
    gdspath = component_path / (component_name + ".gds")

    if not os.path.isfile(gdspath):
        raise ValueError("cannot load `{}`".format(gdspath))

    return gdspath


def load_component(
    component_name: str,
    component_path: PosixPath = CONFIG["gdslib"],
    with_info_labels: bool = True,
    overwrite_cache: bool = False,
) -> Component:
    """ loads GDS, ports (CSV) and metadata (JSON)
    returns a Device

    Args:
        component_name:
        component_path: libary path
        with_info_labels: can remove labal info
        overwrite_cache
    """
    component_path = pathlib.Path(component_path)

    gdspath = component_path / (component_name + ".gds")
    portspath = gdspath.with_suffix(".ports")
    jsonpath = gdspath.with_suffix(".json")

    if not os.path.isfile(gdspath):
        raise ValueError("cannot load `{}`".format(gdspath))

    c = pp.import_gds(str(gdspath), overwrite_cache=overwrite_cache)
    c.name = component_name

    # Remove info labels if needed
    if not with_info_labels:
        for component in list(c.get_dependencies(recursive=True)) + [c]:
            old_label = [
                label
                for label in component.labels
                if label.layer == pp.LAYER.INFO_GEO_HASH
            ]
            if len(old_label) > 0:
                for label in old_label:
                    component.labels.remove(label)

    """ add ports """
    try:
        with open(str(portspath), newline="") as csvfile:
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
    except Exception:
        # print(
        #     f"Could not find a port CSV file for {component_name} in {portspath}"
        # )
        # print(
        #     "ports follow (name, x, y, width, angle, layer_gds_type, layer_gds_purpose)"
        # )
        pass
        # raise (e)

    """ add settings """
    try:
        with open(jsonpath) as f:
            data = json.load(f)
        cell_settings = data["cells"][c.name]
        c.settings.update(cell_settings)
    except Exception:
        pass
        print(f"could not load settings for {c.name} in {jsonpath}")
    return c


def _compare_hash():
    component_type = "coupler90"
    c = load_component(component_type)
    c2 = pp.get_component_type(component_type, gap=0.1)
    print(c.hash_geometry())
    print(c2.hash_geometry())
    pp.show(c)


if __name__ == "__main__":
    component_type = "waveguide"
    c = load_component(component_type)
    print(c.settings)
