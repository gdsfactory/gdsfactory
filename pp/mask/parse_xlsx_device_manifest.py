import json
import xlrd
from pp.config import logging


def parse_device_manifest(
    filepath,
    sheet_name_prefix="Device Manifest",
    row_indices=[0, 1, 2, 3],
    key_field_id=2,
):
    """
    Open device manifest file
    Returns {device_id: {attribute: value}}
    """
    book = xlrd.open_workbook(filepath)

    device_manifest_sheet_names = []

    row_indices_attributes = row_indices[:]
    try:
        row_indices_attributes.remove(key_field_id)
    except:
        print(row_indices_attributes)
        raise

    ## Find all device manifest sheets
    device_manifest_sheet_names = [
        s for s in book.sheet_names() if s.startswith(sheet_name_prefix)
    ]

    devices_dict = {}

    index_of_key_field_id = row_indices.index(key_field_id)

    for sheet_name in device_manifest_sheet_names:
        # print(sheet_name)
        device_manifest = book.sheet_by_name(sheet_name)

        rows_gen = device_manifest.get_rows()
        rows = []
        first_row = next(rows_gen)
        col_names = [
            first_row[i].value.replace(" ", "_") for i in row_indices_attributes
        ]
        print(sheet_name)
        print(col_names)
        print()

        for i in range(1000):
            try:
                _row = next(rows_gen)
            except:
                break
            if _row[0].value:
                rows += [[_clean_value(_row[j].value) for j in row_indices]]
                logging.debug(rows[-1][key_field_id])

        _devices_dict = {}
        for row in rows:
            device_id = row.pop(index_of_key_field_id)
            # print(row)
            _devices_dict[device_id] = {_c: _v for _c, _v in zip(col_names, row)}

            devices_dict.update(_devices_dict)
    return devices_dict


def _clean_value(x):
    x = x.strip()
    splitted = x.split(":")
    if len(splitted) == 1:
        return x
    elif len(splitted) == 2:
        return splitted[1]
    else:
        raise ValueError("invalid value {}".format(x))


def inject_device_manifest_data_in_json(
    dm_filepath, json_filepath, dm_parser_settings={}
):

    device_manifest_data = parse_device_manifest(dm_filepath, **dm_parser_settings)

    with open(json_filepath) as f:
        mask_data = json.loads(f.read())

        does = mask_data["does"]
        cells = mask_data["cells"]
        doe_devices = set()
        for doe in does.values():
            doe_devices.update(doe["cells"])

    ## Inject device manifest data into cells
    for cell_name in doe_devices:

        if cell_name not in cells:
            logging.warning(
                "skip reconcile data for cell {} - cell not in cells".format(cell_name)
            )
            continue

        if cell_name not in device_manifest_data:
            logging.warning(
                "skip reconcile data for cell {} - cell not in device manifest".format(
                    cell_name
                )
            )
            continue

        cells[cell_name].update(device_manifest_data[cell_name])

    mask_data["cells"].update(cells)

    with open(json_filepath, "w") as fw:
        fw.write(json.dumps(mask_data, indent=2))


def inject_data_analysis_params_in_json(
    dm_filepath,
    json_filepath,
    dm_parser_settings={
        "sheet_name_prefix": "Analysis - ",
        "row_indices": [4, 6, 7, 8, 9, 10, 11, 12, 13],
        "key_field_id": 4,
    },
    data_analysis_id_from_cell_name=lambda s: s,
):

    device_id_to_analysis_metadata = parse_device_manifest(
        dm_filepath, **dm_parser_settings
    )

    with open(json_filepath) as f:
        mask_data = json.loads(f.read())

        does = mask_data["does"]
        cells = mask_data["cells"]
        doe_devices = set()
        for doe in does.values():
            doe_devices.update(doe["cells"])

    ## Inject device manifest data into cells
    for cell_name in doe_devices:
        cell = cells[cell_name]

        try:
            cell_id = data_analysis_id_from_cell_name(cell_name)
            cell["analysis"] = device_id_to_analysis_metadata[cell_id]
        except:
            print("No match for data analysis on ", cell_name)

    mask_data["cells"].update(cells)

    with open(json_filepath, "w") as fw:
        fw.write(json.dumps(mask_data, indent=2))
