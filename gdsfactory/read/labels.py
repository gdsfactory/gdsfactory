from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import gdsfactory as gf
from gdsfactory.component_layout import _rotate_points
from gdsfactory.name import clean_name
from gdsfactory.typings import LayerSpec, PathType


def read_labels_yaml(
    csvpath: PathType, prefix: Optional[str] = None
) -> Dict[str, DictConfig]:
    """Read labels from csvfile in YAML format."""
    labels = pd.read_csv(csvpath)
    cells = OmegaConf.create()

    for label in labels.iterrows():
        label = label[1]
        if prefix and not label[0].startswith(prefix):
            continue

        d = OmegaConf.create(label[0])
        x = label[1]
        y = label[2]
        rotation = label[3]
        cell_name = d["component_name"]
        instance_name = clean_name(f"{cell_name}_{x}_{y}")
        # wavelength = d.get("wavelength", 1.55)
        # polarization = d["polarization"]
        # ports = d["ports"]

        d["x"] = x
        d["y"] = y
        d["rotation"] = rotation
        cells[instance_name] = d
    return cells


def add_port_markers(
    gdspath, csvpath, marker_size: int = 20, marker_layer: LayerSpec = (203, 0)
):
    """Add port markers from port info extracted from a gdspath and csvpath."""
    c = gf.Component("overlay")
    c << gf.import_gds(gdspath)
    cells = read_labels_yaml(csvpath=csvpath)

    for cell in cells.values():
        for port in cell["ports"].values():
            r = c << gf.components.rectangle(
                size=(marker_size, marker_size), layer=marker_layer
            )

            x = port["center"][0]
            y = port["center"][1]
            rotation = cell["rotation"]
            x, y = np.array(_rotate_points((x, y), angle=rotation)).flatten()

            r.x = x + cell["x"]
            r.y = y + cell["y"]
    return c


if __name__ == "__main__":
    # c = read_labels_yaml(csvpath="/home/jmatres/mask.csv", prefix="component_name")
    # csvpath = "/home/jmatres/mask.csv"
    # prefix = "component_name"

    # labels = pd.read_csv(csvpath)
    # cells = OmegaConf.create()

    # for label in labels.iterrows():
    #     label = label[1]
    #     if prefix and not label[0].startswith(prefix):
    #         continue

    #     d = OmegaConf.create(label[0])
    #     x = label[1]
    #     y = label[2]
    #     rotation = label[3]
    #     cell_name = d["component_name"]
    #     wavelength = d.get("wavelength", 1.55)
    #     polarization = d["polarization"]
    #     ports = d["ports"]
    #     instance_name = clean_name(f"{cell_name}_{x}_{y}")
    #     cells[instance_name] = d

    csvpath = "/home/jmatres/mask.csv"
    gdspath = "/home/jmatres/mask.gds"
    marker_size = 40
    marker_layer = (203, 0)

    c = gf.Component("overlay")
    c << gf.import_gds(gdspath)
    cells = read_labels_yaml(csvpath=csvpath)

    for cell in cells.values():
        for port in cell["ports"].values():
            r = c << gf.components.rectangle(
                size=(marker_size, marker_size), layer=marker_layer
            )

            x = port["center"][0]
            y = port["center"][1]
            rotation = cell["rotation"]
            x, y = np.array(_rotate_points((x, y), angle=rotation)).flatten()

            r.x = x + cell["x"]
            r.y = y + cell["y"]

    c.show(show_ports=True)
