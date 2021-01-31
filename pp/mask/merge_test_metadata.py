"""Merge mask metadata with test and data analysis protocols

config.yml

```yaml
test_protocols:
    passive_optical_te_coarse:
        wl_min:
        wl_max:
        wl_step:
        polarization: te

    passive_optical_tm_coarse:
        wl_min:
        wl_max:
        wl_step:
        polarization: tm
    ...

```


does.yml

```yaml
doe01:
    instances:
        - cell_name1, x1, y1
        - cell_name2, x2, y2
        - cell_name3, x3, y3

    test_protocols:
        - passive_optical_te_coarse

doe02:
    instances:
        - cell_name21, x21, y21
        - cell_name22, x22, y22
        - cell_name23, x23, y23

    test_protocols:
        - passive_optical_te_coarse
    ...
```
"""

import json
import pathlib
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pp.config import CONFIG


def parse_csv_data(csv_labels_path: Path) -> List[List[str]]:
    with open(csv_labels_path) as f:
        # Get all lines
        lines = [line.replace("\n", "") for line in f.readlines()]

        # Ignore labels for metrology structures
        lines = [line for line in lines if not line.startswith("METR_")]

        # Split lines in fields
        lines = [line.split(",") for line in lines]

        lines = [[s.strip() for s in splitted if s.strip()] for splitted in lines]

        # Remove empty lines
        lines = [line for line in lines if line]
    return lines


def get_cell_from_label(label: str) -> str:
    """get cell name from the label (cell_name is in parenthesis)"""
    cell_name = label.split("(")[1].split(")")[0]
    if cell_name.startswith("loopback"):
        cell_name = "_".join(cell_name.split("_")[1:])
    return cell_name


def load_json(filepath: Path,) -> Dict[str, Any]:
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_yaml(filepath: Path):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data


def merge_test_metadata(
    gdspath: Path = CONFIG["mask_gds"], labels_prefix: str = "opt"
) -> Dict[str, Any]:
    """from a gds mask combines test_protocols and labels positions for each DOE
    Do a map cell: does
    Usually each cell will have only one DOE. But in general it should be allowed for a cell to belong to multiple DOEs

    Args:
        gdspath
        labels_prefix

    Returns:
        saves json file with merged metadata

    """
    gdspath = pathlib.Path(gdspath)
    mask_json_path = gdspath.with_suffix(".json")
    csv_labels_path = gdspath.with_suffix(".csv")
    output_tm_path = gdspath.with_suffix(".tp.json")

    assert mask_json_path.exists(), f"missing mask JSON metadata {mask_json_path}"
    assert csv_labels_path.exists(), f"missing CSV labels {csv_labels_path}"

    metadata = load_json(mask_json_path)
    labels_list = parse_csv_data(csv_labels_path)

    does = metadata.pop("does")
    cells = metadata.pop("cells")

    c = {}

    for label, x, y in labels_list:
        cell = get_cell_from_label(label)
        c[cell] = c.get(cell, dict())
        c[cell][label] = dict(x=x, y=y)

    d = dict(cells_to_test=c, metadata=metadata, does=does, cells=cells)

    with open(output_tm_path, "w") as json_out:
        json.dump(d, json_out, indent=2)

    return metadata


if __name__ == "__main__":
    from pp import CONFIG

    gdspath = CONFIG["repo_path"] / "samples" / "mask" / "build" / "mask" / "mask.gds"
    d = merge_test_metadata(gdspath)
    print(d)
