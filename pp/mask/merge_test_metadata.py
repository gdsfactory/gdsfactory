""" merges mask metadata with test and data analysis protocols


mask_tm.json

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

does:
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
"""

import os
import json
import yaml


def parse_csv_data(csv_labels_path):
    with open(csv_labels_path) as f:
        # Get all lines
        lines = [line.replace("\n", "") for line in f.readlines()]

        # Ignore labels for metrology structures
        lines = [line for line in lines if not line.startswith("METR_")]

        # Split lines in fields
        lines = [line.split(",") for line in lines]

        lines = [[s.strip() for s in splitted if s.strip()] for splitted in lines]

        # Remove empty lines
        lines = [l for l in lines if l]
    return lines


def get_cell_from_label(label):
    return label.split("(")[1].split(")")[0]


def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_yaml(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data


def merge_test_metadata(gdspath):
    """ from a gds mask combines test_protocols and labels positions for each DOE
    Do a map cell: does
    Usually each cell will have only one DOE. But in general it should be allowed for a cell to belong to multiple DOEs

    Args:
        gdspath

    Returns:
        saves json file with merged metadata
        
    """
    mask_json_path = gdspath.with_suffix(".json")
    csv_labels_path = gdspath.with_suffix(".csv")
    output_tm_path = gdspath.with_suffix(".tp.json")
    tm_dict = {}

    mask_directory = gdspath.parent
    mask_build_directory = mask_directory.parent
    mask_root_directory = mask_build_directory.parent
    test_protocols_path = mask_root_directory / "test_protocols.yml"
    analysis_protocols_path = mask_root_directory / "data_analysis_protocols.yml"

    assert os.path.isfile(mask_json_path), "missing {}".format(mask_json_path)
    assert os.path.isfile(csv_labels_path), "missing {}".format(csv_labels_path)

    # mask_data = json.loads(open(mask_json_path).read())
    # mask_data = hiyapyco.load(mask_json_path)
    mask_data = load_json(mask_json_path)

    if os.path.isfile(test_protocols_path):
        test_protocols = load_yaml(test_protocols_path)
        tm_dict["test_protocols"] = test_protocols
    if os.path.isfile(analysis_protocols_path):
        analysis_protocols = load_yaml(analysis_protocols_path)
        tm_dict["analysis_protocols"] = analysis_protocols

    data = parse_csv_data(csv_labels_path)
    # cell_x_y = [(get_cell_from_label(l), x, y) for l, x, y in data]

    does = mask_data["does"]
    cells = mask_data["cells"]

    cell_to_does = {}
    for doe_name, doe in does.items():
        for c in doe["cells"]:
            if c not in cell_to_does:
                cell_to_does[c] = set()
            cell_to_does[c].update([doe_name])

    tm_dict["does"] = {}
    doe_tm = tm_dict["does"]
    doe_tm.update(does)
    for doe_name, doe in doe_tm.items():
        doe.pop("cells")
        doe["instances"] = {}

    ## Cell instances which need to be measured MUST have a unique cell name
    for label, x, y in data:
        cell_name = get_cell_from_label(label)
        if cell_name not in cell_to_does:
            continue
        cell_does = cell_to_does[cell_name]
        for doe_name in cell_does:
            _doe = doe_tm[doe_name]

            if cell_name not in _doe["instances"]:
                # Unique Cell instance to labels and coordinates
                _doe["instances"][cell_name] = []
            _doe["instances"][cell_name].append({"label": label, "x": x, "y": y})

    # Adding the cells settings
    tm_dict["cells"] = cells

    with open(output_tm_path, "w") as json_out:
        json.dump(tm_dict, json_out, indent=2)


if __name__ == "__main__":
    from sample_mask.config import CONFIG

    gdspath = CONFIG["repo_path"] / "build" / "mask" / "sample_mask.gds"
    merge_test_metadata(gdspath)
