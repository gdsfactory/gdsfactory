""" gdspy based placer

YAML defines component DOE settings and placement

.. code:: yaml

    placer:
        width: 30000  # um
        height: 20000 # um
        x_spacing: 1000 # x spacing origin to origin between the components
        y_spacing: 500 #  y spacing origin to origin between the components
        x_start: 500
        y_start: 100

    mmi1x2_gap:
        doe_name: doe1
        component: mmi1x2
        gap: [0.5, 0.6]
        length: 10

    mmi1x2:
        doe_name: doe1
        component: mmi1x2
        length: [11, 12]
        gap: [0.2, 0.3]
        do_permutation: False

    placement:
        A-B1-2: doe1
"""

import os
import pathlib
import sys

from omegaconf import OmegaConf

import pp
from pp.components import component_factory
from pp.config import CONFIG
from pp.doe import get_settings_list, load_does
from pp.write_component import write_component_report, write_gds


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def placer_grid_cell_refs(
    component_factory, cols=1, rows=1, dx=10.0, dy=10.0, x0=0, y0=0, **settings
):
    if callable(component_factory):
        settings_list = get_settings_list(**settings)
        component_list = [component_factory(**s) for s in settings_list]
    else:
        component_list = component_factory

    indices = [(i, j) for j in range(cols) for i in range(rows)]

    if rows * cols < len(component_list):
        raise ValueError(
            "Shape ({}, {}): Not enough emplacements ({}) for all these components ({}).".format(
                rows, cols, len(indices), len(component_list)
            )
        )
    components = []
    for component, (i, j) in zip(component_list, indices):
        c_ref = component.ref(position=(x0 + j * dx, y0 + i * dy))
        components += [c_ref]

    return components


def pack_horizontal(
    cells, row_ids=None, x0=0, y0=0, align_x="W", align_y="S", margin_x=20, margin_y=20,
):
    """
    Args:
        cells: a list of cells  (size n)
        row_ids: a list of row ids (size n)
            where each id represents the row where the cell should be placed
            None by default => all cells in the same row

    returns a list of cell references
    """
    heights = [c.size_info.height for c in cells]

    row_ids = row_ids or [0] * len(cells)

    if len(cells) != len(row_ids):
        raise ValueError(
            "Each cell should be assigned a row id. \
        Got {} cells for {} row ids".format(
                len(cells), len(row_ids)
            )
        )

    # Find the height of each row to fit the cells
    # Also group the cells by row

    unique_row_ids = list(set(row_ids))
    unique_row_ids.sort()
    _row_to_heights = {r: [] for r in set(row_ids)}
    row_to_cells = {r: [] for r in unique_row_ids}
    for row, h, cell in zip(row_ids, heights, cells):
        _row_to_heights[row] += [h]
        row_to_cells[row] += [cell]

    row_to_height = {k: max(v) for k, v in _row_to_heights.items()}

    components = []

    # Do the packing per row
    y = y0
    for row in unique_row_ids:
        cells = row_to_cells[row]
        x = x0

        for c in cells:
            if align_x == "W" and align_y == "S":
                component_origin = c.size_info.sw
            elif align_x == "E" and align_y == "S":
                component_origin = c.size_info.se
            elif align_x == "E" and align_y == "N":
                component_origin = c.size_info.ne
            elif align_x == "W" and align_y == "N":
                component_origin = c.size_info.nw
            try:
                components += [c.ref(position=-component_origin + (x, y))]
            except ValueError as e:
                if align_x not in ["W", "E"]:
                    print("align_x should be `W`, `E` or a float")
                if align_y not in ["N", "S"]:
                    print("align_y should be `N`, `S` or a float")
                raise e

            if align_x == "W":
                x += c.size_info.width + margin_x
            else:
                x += -c.size_info.width - margin_x

        if align_y == "S":
            y += row_to_height[row] + margin_y
        else:
            y += -row_to_height[row] - margin_y

    return components


def pack_vertical(
    cells, col_ids=None, x0=0, y0=0, align_x="W", align_y="S", margin_x=20, margin_y=20,
):
    """
    Args:
        cells: a list of cells  (size n)
        col_ids: a list of column ids (size n)
            where each id represents the row where the cell should be placed
            None by default => all cells are packed in the same column

    returns a list of cell references
    """
    widths = [c.size_info.width for c in cells]
    col_ids = col_ids or [0] * len(cells)

    if len(cells) != len(col_ids):
        raise ValueError(
            "Each cell should be assigned a row id. \
        Got {} cells for {} col ids".format(
                len(cells), len(col_ids)
            )
        )

    # Find the width of each column to fit the cells
    # Also group the cells by column

    unique_col_ids = list(set(col_ids))
    unique_col_ids.sort()
    _col_to_widths = {r: [] for r in set(col_ids)}
    col_to_cells = {r: [] for r in unique_col_ids}
    for col, w, cell in zip(col_ids, widths, cells):
        _col_to_widths[col] += [w]
        col_to_cells[col] += [cell]

    col_to_width = {k: max(v) for k, v in _col_to_widths.items()}

    components = []

    # Do the packing per column
    x = x0
    for col in unique_col_ids:
        cells = col_to_cells[col]
        y = y0

        for c in cells:
            if align_x == "W" and align_y == "S":
                component_origin = c.size_info.sw
            elif align_x == "E" and align_y == "S":
                component_origin = c.size_info.se
            elif align_x == "E" and align_y == "N":
                component_origin = c.size_info.ne
            elif align_x == "W" and align_y == "N":
                component_origin = c.size_info.nw
            components += [c.ref(position=-component_origin + (x, y))]

            if align_y == "S":
                y += c.size_info.height + margin_y
            else:
                y += -c.size_info.height - margin_y

        if align_x == "W":
            x += col_to_width[col] + margin_x
        else:
            x += -col_to_width[col] - margin_x

    return components


def placer_fixed_coords(cells, x, y, **kwargs):
    if len(cells) == 1:
        cells = cells * len(x)

    return [c.ref(position=(_x, _y)) for c, _x, _y in zip(cells, x, y)]


PLACER_NAME2FUNC = {
    "grid": placer_grid_cell_refs,
    "pack_row": pack_horizontal,
    "pack_col": pack_vertical,
    "fixed_coords": placer_fixed_coords,
}


def load_placer_with_does(filepath, defaults=None):
    """load placer settings

    Args:
        filepath: a yaml file containing the does and placer information

    Returns:
        a dictionnary of DOEs with:
        {
            doe_name1: [(component_factory_name, parameters), ...]
            doe_name2: [(component_factory_name, parameters), ...]
            ...
        }

    """
    defaults = defaults or {"do_permutation": True}
    does = {}
    data = OmegaConf.load(filepath)

    placer_info = data.pop("placer")
    component_placement = data.pop("placement")
    gds_files = {}
    if "gds" in data.keys():
        gds_files.update(data.pop("gds"))

    for doe_chunk in data.values():
        doe_name = doe_chunk.pop("doe_name")
        component_type = doe_chunk.pop("component")

        if doe_name not in does:
            does[doe_name] = {"settings": [], "component_type": component_type}

        do_permutation = defaults["do_permutation"]
        if "do_permutation" in doe_chunk:
            do_permutation = doe_chunk.pop("do_permutation")

        doe = does[doe_name]
        # All the remaining parameters are component parameters
        if "settings" in doe_chunk:
            settings = doe_chunk.pop("settings")
        else:
            settings = {}
        doe["list_settings"] += get_settings_list(do_permutation, **settings)

        # check that the doe is valid (only one type of component)
        assert (
            component_type == doe["component_type"]
        ), "There can be only one component type per doe. Got {} while expecting {}".format(
            component_type, doe["component_type"]
        )

    return does, placer_info, component_placement, gds_files


CONTENT_SEP = " , "


def save_doe(
    doe_name, components, doe_root_path=CONFIG["cache_doe_directory"], precision=1e-9
):
    """
    Save all components from this DOE in a tmp cache folder
    """
    doe_dir = pathlib.Path(doe_root_path) / doe_name
    doe_dir.mkdir(parents=True, exist_ok=True)

    # Store list of component names - order matters
    component_names = [c.name for c in components]
    content_file = doe_dir / "content.txt"
    with open(content_file, "w") as fw:
        fw.write(CONTENT_SEP.join(component_names))

    for c in components:
        gdspath = doe_dir / f"{c.name}.gds"
        write_gds(c, gdspath=gdspath, precision=precision)
        write_component_report(c, json_path=gdspath.with_suffix(".json"))


def load_doe_from_cache(doe_name, doe_root_path=None):
    """
    Load all components for this DOE from the cache
    """
    if doe_root_path is None:
        doe_root_path = CONFIG["cache_doe_directory"]
    doe_dir = os.path.join(doe_root_path, doe_name)
    content_file = os.path.join(doe_dir, "content.txt")
    with open(content_file) as f:
        component_names = f.read().split(CONTENT_SEP)

    gdspaths = [os.path.join(doe_dir, name + ".gds") for name in component_names]
    components = [pp.import_gds(gdspath) for gdspath in gdspaths]
    return components


def load_doe_component_names(doe_name, doe_root_path=None):
    if doe_root_path is None:
        doe_root_path = CONFIG["cache_doe_directory"]
    doe_dir = os.path.join(doe_root_path, doe_name)
    content_file = os.path.join(doe_dir, "content.txt")

    with open(content_file) as f:
        component_names = f.read().split(CONTENT_SEP)
    return component_names


def doe_exists(doe_name, list_settings, doe_root_path=None):
    """
    Check whether the folder exists and that the number of items in content.txt
    matches the number of items in list_settings
    """
    if doe_root_path is None:
        doe_root_path = CONFIG["cache_doe_directory"]
    doe_dir = os.path.join(doe_root_path, doe_name)
    content_file = os.path.join(doe_dir, "content.txt")
    if not os.path.exists(content_file):
        return False
    with open(content_file) as f:
        component_names = f.read().split(CONTENT_SEP)

    if len(component_names) == len(list_settings) or (
        len(list_settings) == 0 and len(component_names) == 1
    ):
        return True

    print(
        "doe_exists - DOE",
        doe_name,
        "needs regeneration",
        len(component_names),
        len(list_settings),
    )
    return False


def component_grid_from_yaml(filepath, precision=1e-9):
    """Returns a Component composed of DOEs/components given in a yaml file
    allows for each DOE to have its own x and y spacing (more flexible than method1)
    """
    input_does = OmegaConf.load(str(filepath))
    mask_settings = input_does["mask"]
    does = load_does(filepath)

    placed_doe = None
    placed_does = {}
    if mask_settings.get("name"):
        component_grid = pp.Component(mask_settings["name"])
    else:
        component_grid = pp.Component()

    default_cache_enabled = (
        mask_settings["cache_enabled"] if "cache_enabled" in mask_settings else False
    )
    for doe_name, doe in does.items():
        list_settings = doe["settings"]
        component_type = doe["component"]

        # description = doe["description"] if "description" in doe else ''
        # test = doe["test"] if "test" in doe else {}
        # analysis = doe["analysis"] if "analysis" in doe else {}

        # Get DOE policy concerning the cache
        cache_enabled = (
            doe["cache_enabled"] if "cache_enabled" in doe else default_cache_enabled
        )

        components = None

        # If cache enabled, attempt to load from cache
        if cache_enabled:
            try:
                components = load_doe_from_cache(doe_name)
            except Exception as e:
                _print(e)
                components = None

        # If no component is loaded, build them
        if components is None:
            print("{} - Generating components...".format(doe_name))
            components = build_components(component_type, list_settings)

            # After building the components, if cache enabled, save them
            if cache_enabled:
                save_doe(doe_name, components, precision=precision)
        else:
            _print("{} - Loaded components from cache".format(doe_name))

        # _print(doe_name, [c.name for c in components])
        # Find placer information

        default_settings = {"align_x": "W", "align_y": "S", "margin": 10}

        if "placer" in doe:
            placer_type = doe["placer"].pop("type")
            _placer = PLACER_NAME2FUNC[placer_type]

            # All other attributes are assumed to be settings for the placer
            settings = default_settings.copy()
            settings.update(doe["placer"])

            # x0, y0 can either be float or string
            x0 = settings.pop("x0")
            y0 = settings.pop("y0")

            # Check whether we are doing relative or absolute placement
            if (x0 in ["E", "W"] or y0 in ["N", "S"]) and not placed_doe:
                raise ValueError(
                    "At least one DOE must be placed to use\
                relative placement"
                )

            # For relative placement (to previous DOE)
            if "margin_x" not in settings:
                settings["margin_x"] = settings["margin"]
            if "margin_y" not in settings:
                settings["margin_y"] = settings["margin"]

            margin_x = settings["margin_x"]
            margin_y = settings["margin_y"]
            align_x = settings["align_x"]
            align_y = settings["align_y"]

            # Making sure that the alignment is sensible depending on how we stack

            # If we specify a DOE to place next to, use it
            if "next_to" in settings:
                placed_doe = placed_does[settings.pop("next_to")]

            # Otherwise, use previously placed DOE as starting point
            if x0 == "E":
                x0 = placed_doe.size_info.east
                if align_x == "W":
                    x0 += margin_x

            if x0 == "W":
                x0 = placed_doe.size_info.west
                if align_x == "E":
                    x0 -= margin_x

            if y0 == "N":
                y0 = placed_doe.size_info.north
                if align_y == "S":
                    y0 += margin_y

            if y0 == "S":
                y0 = placed_doe.size_info.south
                if align_y == "N":
                    y0 -= margin_y

            # Add x0, y0 in settings as float
            settings["x0"] = x0
            settings["y0"] = y0

            placed_components = _placer(components, **settings)
        else:
            # If no placer is specified, we assume this is a grid placer
            cols, rows = doe["shape"]
            x0, y0 = doe["origin"]
            dx, dy = doe["spacing"]

            placed_components = placer_grid_cell_refs(
                components, cols, rows, dx, dy, x0=x0, y0=y0
            )

        # Place components within a cell having the DOE name
        placed_doe = pp.Component()
        placed_doe.add(placed_components)
        placed_doe.name = doe_name
        placed_does[doe_name] = placed_doe

        # # Write the json and md metadata / report
        # write_doe_metadata(
        # doe_name=doe_name,
        # cells = placed_components,
        # list_settings=list_settings,
        # flag_write_component=False,
        # description=description,
        # test=test,
        # analysis=analysis
        # )

        component_grid.add_ref(placed_doe)

    return component_grid


def build_components(
    component_type, list_settings, component_factory=component_factory
):
    components = []

    # If no settings passed, generate a single component with defaults
    if not list_settings:
        component_function = component_factory[component_type]
        component = component_function()
        components += [component]
        return components

    for settings in list_settings:
        component_function = component_factory[component_type]
        component = component_function(**settings)
        components += [component]

    return components


if __name__ == "__main__":
    pass
