import os
import json

import hiyapyco
from pp.mask.parse_xlsx_device_manifest import parse_device_manifest


def parse_mask_json(mask_json_path):
    with open(mask_json_path) as f:
        data = json.loads(f.read())

    ordered_fields = ["name", "json_version", "width", "height"]
    ordered_fields += ["git_hash_autoplacer", "git_hash_mask", "git_hash_pp"]
    ordered_fields += ["description", "does", "cells"]

    return {k: data[k] for k in ordered_fields if k in data}


def parse_csv_data(csv_labels_path):
    with open(csv_labels_path) as f:
        lines = [line.replace("\n", "").split(",") for line in f.readlines()]
        lines = [[s.strip() for s in splitted if s.strip()] for splitted in lines]
        lines = [l for l in lines if l]
    return lines


def get_cell_from_label(label):
    return label.split("(")[1].split(")")[0]


def merge_all_metadata(gdspath, device_manifest_path):
    """ from a gds mask combines test_protocols and labels positions for each DOE
    Do a map cell: does
    Usually each cell will have only one DOE. But in general it should be allowed for a cell to belong to multiple DOEs
    """
    mask_json_path = gdspath.replace(".gds", ".json")
    csv_labels_path = gdspath.replace(".gds", ".csv")
    output_tm_path = gdspath.replace(".gds", ".tp.json")
    tm_dict = {}

    device_manifest_data = parse_device_manifest(device_manifest_path)

    mask_directory = os.path.split(gdspath)[0]
    mask_build_directory = os.path.split(mask_directory)[0]
    mask_root_directory = os.path.split(mask_build_directory)[0]
    test_protocols_path = os.path.join(mask_root_directory, "test_protocols.yml")
    analysis_protocols_path = os.path.join(
        mask_root_directory, "data_analysis_protocols.yml"
    )

    assert os.path.isfile(mask_json_path), "missing {}".format(mask_json_path)
    assert os.path.isfile(csv_labels_path), "missing {}".format(csv_labels_path)

    mask_data = parse_mask_json(mask_json_path)

    if os.path.isfile(test_protocols_path):
        test_protocols = hiyapyco.load(test_protocols_path)
        tm_dict["test_protocols"] = test_protocols
    if os.path.isfile(analysis_protocols_path):
        analysis_protocols = hiyapyco.load(analysis_protocols_path)
        tm_dict["analysis_protocols"] = analysis_protocols

    data = parse_csv_data(csv_labels_path)
    # cell_x_y = [(get_cell_from_label(l), x, y) for l, x, y in data]

    does = mask_data["does"]
    cells = mask_data["cells"]
    doe_devices = set()
    for doe in does.values():
        doe_devices.update(doe["cells"])

    # from pprint import pprint
    # pprint(list(cells.keys()))
    # pprint(list(device_manifest_data.keys()))

    ## Inject device manifest data into cells
    for cell_name in doe_devices:

        if cell_name not in cells:
            print(
                "skip reconcile data for cell {} - cell not in cells".format(cell_name)
            )
            continue

        if cell_name not in device_manifest_data:
            print(
                "skip reconcile data for cell {} - cell not in device manifest".format(
                    cell_name
                )
            )
            continue
        cells[cell_name].update(device_manifest_data[cell_name])

    ## Map cell to DOEs - generic case where a cell could belong to multiple experiments
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
    import pp

    gdspath = os.path.join(
        pp.CONFIG["masks_path"], "sample", "build", "mask", "sample.gds"
    )
    merge_all_metadata(gdspath)
