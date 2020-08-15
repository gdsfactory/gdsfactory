"""
you can define both DOEs and placer information in a YAML file
all the placer information need to be nested inside a placer section

iso_lines_coarse1:
  component: ISO_COARS_OPT1
  settings:
    dx: [50.]

  placer:
    type: pack_row/ pack_col/ grid / fixed_coords
        pack_grid:
    x0: 0
    y0: 0
    align_x: W
    align_y: S
    next_to: iso_lines_coarse1

"""


import os
import sys
import collections
import numpy as np
from omegaconf import OmegaConf
import klayout.db as pya

import pp.autoplacer.text as text
from pp.autoplacer.helpers import import_cell, load_gds, CELLS
from pp.config import CONFIG

UM_TO_GRID = 1e3
DEFAULT_BBOX_LAYER_IGNORE = [(8484, 8484)]


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def to_grid(x, um_to_grid=UM_TO_GRID):
    return int(x * um_to_grid)


class SizeInfo:
    def __init__(
        self,
        cell,
        layout=None,
        ignore_layers=DEFAULT_BBOX_LAYER_IGNORE,
        um_to_grid=UM_TO_GRID,
    ):
        """
        layout is required if cell is a cell reference instead of a cell
        """
        if isinstance(cell, pya.CellInstArray):
            bbox = cell.bbox(layout)
        else:
            parent_layout = cell.layout()
            layer_indexes = parent_layout.layer_indexes()

            for process_id, purpose_id in ignore_layers:
                layer_id = parent_layout.find_layer(process_id, purpose_id)
                if layer_id in layer_indexes:
                    layer_indexes.remove(layer_id)

            bbox = None
            for layer_id in layer_indexes:
                per_layer_bbox = cell.bbox_per_layer(layer_id)
                if bbox is None:
                    bbox = per_layer_bbox
                else:
                    bbox = bbox + per_layer_bbox

        self.box = bbox

        self.west = self.box.left / um_to_grid
        self.east = self.box.right / um_to_grid

        self.south = self.box.bottom / um_to_grid
        self.north = self.box.top / um_to_grid

        self.width = self.east - self.west
        self.height = self.north - self.south

        xc = int(0.5 * (self.east + self.west))
        yc = int(0.5 * (self.north + self.south))

        self.sw = np.array([self.west, self.south])
        self.se = np.array([self.east, self.south])
        self.nw = np.array([self.west, self.north])
        self.ne = np.array([self.east, self.north])

        self.cw = np.array([self.west, yc])
        self.ce = np.array([self.east, yc])
        self.nc = np.array([xc, self.north])
        self.sc = np.array([xc, self.south])
        self.cc = self.center = np.array([xc, yc])

    @property
    def rect(self):
        w, e, s, n = self.west, self.east, self.south, self.north
        return [(w, s), (e, s), (e, n), (w, n)]

    def __str__(self):
        return "w: {}\ne: {}\ns: {}\nn: {}\n".format(
            self.west, self.east, self.south, self.north
        )


def placer_grid_cell_refs(
    cells,
    cols=1,
    rows=1,
    dx=10.0,
    dy=10.0,
    x0=0,
    y0=0,
    um_to_grid=UM_TO_GRID,
    **settings,
):
    """cells: list of cells - order matters for placing"""

    indices = [(i, j) for j in range(cols) for i in range(rows)]

    if rows * cols < len(cells):
        raise ValueError(
            "Shape ({}, {}): Not enough emplacements ({}) for all these components"
            " ({}).".format(rows, cols, len(indices), len(cells))
        )
    components = []
    for cell, (i, j) in zip(cells, indices):
        _x = int((x0 + j * dx) * um_to_grid)
        _y = int((y0 + i * dy) * um_to_grid)

        transform = pya.Trans(_x, _y)
        c_ref = pya.CellInstArray(cell.cell_index(), transform)
        components += [c_ref]

    return components


def pack_row(
    cells,
    row_ids=None,
    nb_cols=None,
    x0=0,
    y0=0,
    align_x="W",
    align_y="S",
    margin=20,
    margin_x=None,
    margin_y=None,
    um_to_grid=UM_TO_GRID,
    period_x=None,
    period_y=None,
    rotation=0,
):
    """
    Args:
        cells: a list of cells  (size n)
        row_ids: a list of row ids (size n)
            where each id represents the row where the cell should be placed
            None by default => all cells in the same row


        period_x, period_y: not used by default,
            if set, use this period instead of computing the component spacing
            from the margin and the component dimension


    returns a list of cell references
    """
    si_list = [SizeInfo(c, um_to_grid=um_to_grid) for c in cells]
    heights = [si.height for si in si_list]
    margin_y = margin_y if margin_y is not None else margin
    margin_x = margin_x if margin_x is not None else margin

    if row_ids is None:
        row_ids = []
        nb_cells = len(cells)
        if nb_cols is None:
            nb_cols = len(cells)
        nb_full_rows = nb_cells // nb_cols
        nb_cols_last_row = nb_cells % nb_cols
        for row_id in range(nb_full_rows):
            row_ids += [row_id] * nb_cols

        last_row_index = row_id + 1
        row_ids += [last_row_index] * nb_cols_last_row

    if len(cells) != len(row_ids):
        raise ValueError(
            "Each cell should be assigned a row id.         Got {} cells for {} row ids".format(
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
            si = SizeInfo(c, um_to_grid=um_to_grid)
            if align_x == "W" and align_y == "S":
                component_origin = si.sw
            elif align_x == "E" and align_y == "S":
                component_origin = si.se
            elif align_x == "E" and align_y == "N":
                component_origin = si.ne
            elif align_x == "W" and align_y == "N":
                component_origin = si.nw
            try:
                _x = to_grid(x - component_origin[0], um_to_grid)
                _y = to_grid(y - component_origin[1], um_to_grid)

                transform = pya.Trans(rotation / 2, 0, _x, _y)
                # transform = pya.Trans(_x, _y)
                c_ref = pya.CellInstArray(c.cell_index(), transform)
                components += [c_ref]

            except:
                print(x, component_origin[0], um_to_grid)
                print("ISSUE PLACING AT", _x, _y)
                if align_x not in ["W", "E"]:
                    _print("align_x should be `W`, `E` or a float")
                if align_y not in ["N", "S"]:
                    _print("align_y should be `N`, `S` or a float")
                # raise

            dx = si.width + margin_x if period_x is None else period_x
            if align_x == "W":
                x += dx
            else:
                x -= dx

        dy = row_to_height[row] + margin_y if period_y is None else period_y

        if align_y == "S":
            y += dy
        else:
            y -= dy

    return components


def pack_col(
    cells,
    col_ids=None,
    nb_rows=None,
    x0=0,
    y0=0,
    align_x="W",
    align_y="S",
    margin=20,
    margin_x=None,
    margin_y=None,
    um_to_grid=UM_TO_GRID,
    period_x=None,
    period_y=None,
):
    """
    Args:
        cells: a list of cells  (size n)
        col_ids: a list of column ids (size n)
            where each id represents the row where the cell should be placed
            None by default => all cells are packed in the same column

    returns a list of cell references
    """
    widths = [SizeInfo(c, um_to_grid=um_to_grid).width for c in cells]
    margin_y = margin_y if margin_y is not None else margin
    margin_x = margin_x if margin_x is not None else margin

    if col_ids is None:
        col_ids = []
        nb_cells = len(cells)
        if nb_rows is None:
            nb_rows = len(cells)
        nb_full_cols = nb_cells // nb_rows
        nb_rows_last_col = nb_cells % nb_rows
        for col_id in range(nb_full_cols):
            col_ids += [col_id] * nb_rows

        last_col_index = col_id + 1
        col_ids += [last_col_index] * nb_rows_last_col

    if len(cells) != len(col_ids):
        raise ValueError(
            "Each cell should be assigned a row id.         Got {} cells for {} col ids".format(
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
            si = SizeInfo(c, um_to_grid=um_to_grid)
            if align_x == "W" and align_y == "S":
                component_origin = si.sw
            elif align_x == "E" and align_y == "S":
                component_origin = si.se
            elif align_x == "E" and align_y == "N":
                component_origin = si.ne
            elif align_x == "W" and align_y == "N":
                component_origin = si.nw

            _x = to_grid(x - component_origin[0], um_to_grid=um_to_grid)
            _y = to_grid(y - component_origin[1], um_to_grid=um_to_grid)

            try:
                transform = pya.Trans(_x, _y)
                c_ref = pya.CellInstArray(c.cell_index(), transform)
                components += [c_ref]
            except:
                print(x, component_origin[0], um_to_grid)
                print(y, component_origin[1], um_to_grid)
                print("ISSUE PLACING AT", _x, _y)
                print("ISSUE PLACING at", _x, _y)

            dy = si.height + margin_y if period_y is None else period_y
            if align_y == "S":
                y += dy
            else:
                y -= dy

        dx = col_to_width[col] + margin_x if period_x is None else period_x
        if align_x == "W":
            x += dx
        else:
            x -= dx

    return components


def placer_fixed_coords(
    cells, x, y, x0=0, y0=0, do_permutation=False, um_to_grid=UM_TO_GRID, **kwargs
):
    """place cells using a list of coordinates"""

    # List all coordinates
    if do_permutation:
        coords = [(_x, _y) for _x in x for _y in y]
    else:
        coords = [(_x, _y) for _x, _y in zip(x, y)]

    # Update origin
    coords = [(c[0] + x0, c[1] + y0) for c in coords]

    # Generate cell list
    if len(cells) == 1:
        cells = cells * len(coords)

    # update coordinates from um to grid
    coords = [(int(c[0] * um_to_grid), int(c[1] * um_to_grid)) for c in coords]

    # Generate transforms
    transforms = [pya.Trans(*c) for c in coords]

    return [pya.CellInstArray(c.cell_index(), t) for c, t in zip(cells, transforms)]


def load_yaml(filepath, defaults={"do_permutation": True}):
    """load placer settings

    Args:
        filepath: a yaml file containing the does and placer information

    Returns:
        a dictionnary of DOEs with:
        {
            doe_name1: {...}
            doe_name2: {...}
            ...
        }

    """

    does = {}
    data = OmegaConf.load(filepath)
    data = OmegaConf.to_container(data)
    mask = data.pop("mask")

    if "layer_doe_label" not in mask:
        mask["layer_doe_label"] = (102, 6)

    for doe_name, doe in data.items():
        # do_permutation = defaults["do_permutation"]
        # if "do_permutation" in doe:
        # do_permutation = doe.pop("do_permutation")
        _doe = {}
        _doe.update(doe)
        does[doe_name] = _doe
    return does, mask


DOE_CELLS = {}


def load_doe(doe_name, doe_root):
    """
    Load all components for this DOE from the cache
    """
    doe_dir = os.path.join(doe_root, doe_name)
    content_file = os.path.join(doe_dir, "content.txt")

    if os.path.isfile(content_file):
        with open(content_file) as f:
            lines = f.read().split("\n")
            line = lines[0]

            if line.startswith("TEMPLATE:"):
                """
                If using a template, load the GDS from DOE folder used as a template
                """
                template_name = line.split(":")[1].strip()
                return load_doe(template_name, doe_root)

            else:
                """
                Otherwise load the GDS from the current folder
                """
                component_names = line.split(" , ")
                gdspaths = [
                    os.path.join(doe_dir, name + ".gds") for name in component_names
                ]
                cells = [load_gds(gdspath) for gdspath in gdspaths]

        # print("LOAD DOE")
        # for _c in cells:
        # print(_c.top_cell().name)
        # print()

        return cells


PLACER_NAME2FUNC = {
    "grid": placer_grid_cell_refs,
    "pack_row": pack_row,
    "pack_col": pack_col,
    "fixed_coords": placer_fixed_coords,
}


def separate_does_from_templates(dicts):
    templates = {}
    does = {}
    for name, d in dicts.items():
        if "type" in d.keys():  # and d["type"] == "template":
            templates[name] = d
        else:
            does[name] = d

    # We do not want to propagate "type": template to the does => removing it here
    for d in templates.values():
        d.pop("type")

    return does, templates


def update_dicts_recurse(target_dict, default_dict):
    target_dict = target_dict.copy()
    default_dict = default_dict.copy()
    for k, v in default_dict.items():
        if k not in target_dict:
            vtype = type(v)
            if vtype == dict or vtype == collections.OrderedDict:
                target_dict[k] = v.copy()  # To avoid issues when popping
            else:
                target_dict[k] = v
        else:
            vtype = type(target_dict[k])
            if vtype == dict or vtype == collections.OrderedDict:
                target_dict[k] = update_dicts_recurse(target_dict[k], default_dict[k])
    return target_dict


def place_from_yaml(
    filepath_yaml,
    root_does=CONFIG["cache_doe_directory"],
    precision=1e-9,
    fontpath=text.FONT_PATH,
):
    """Returns a gds cell composed of DOEs/components given in a yaml file
    allows for each DOE to have its own x and y spacing (more flexible than method1)

    Args:
        filepath_yaml:
        root_does: used for cache, requires content.txt
    """
    transform_identity = pya.Trans(0, 0)
    dicts, mask_settings = load_yaml(filepath_yaml)

    does, templates = separate_does_from_templates(dicts)

    placed_doe = None
    placed_does = {}
    top_level_name = mask_settings.get("name", "TOP_LEVEL")
    layer_doe_label = mask_settings["layer_doe_label"]
    top_level_layout = pya.Layout()

    # Set database units according to precision
    top_level_layout.dbu = precision / 1e-6
    dbu = top_level_layout.dbu
    um_to_grid = int(1 / dbu)

    top_level = top_level_layout.create_cell(top_level_name)
    global CELLS
    CELLS[top_level_name] = top_level_layout

    default_doe_settings = {
        "add_doe_label": False,
        "add_doe_visual_label": False,
        "dx_visual_label": 0,
        "dy_visual_label": 0,
    }

    for doe_name, doe in does.items():

        # If a template is specified, apply it
        if "template" in doe:
            doe_templates = doe["template"]
            if type(doe_templates) != list:
                doe_templates = [doe_templates]
            for doe_template in doe_templates:
                try:
                    doe = update_dicts_recurse(doe, templates[doe_template])

                except:
                    print(doe_template, "does not exist")
                    raise
        doe = update_dicts_recurse(doe, default_doe_settings)

        # Get all the components
        components = load_doe(doe_name, root_does)

        """
        # Check that the high level components are all unique
        # For now this is mostly to circumvent a bug
        # But the design manual also specifies that DOE components should have
        # unique names. So one instance per cell
        """

        if components:
            if len(components) != len(set([_c.top_cell().name for _c in components])):
                __dict_component_debug = {}
                for _c in components:
                    _name = _c.top_cell().name
                    if _name not in __dict_component_debug:
                        __dict_component_debug[_name] = 0
                    __dict_component_debug[_name] += 1
                duplicates_components = [
                    _name
                    for _name, _count in __dict_component_debug.items()
                    if _count > 1
                ]
                print("Please remove duplicate components at DOE entry level: ")
                print(duplicates_components)

            components = [
                import_cell(top_level_layout, _c.top_cell()) for _c in components
            ]

        # Find placer information
        default_placer_settings = {
            "align_x": "W",
            "align_y": "S",
            "margin": 10,
            "x0": "E",
            "y0": "S",
        }
        settings = default_placer_settings.copy()
        placer = doe.get("placer")

        if placer:
            placer_type = placer.pop("type", "pack_col")
            settings.update(doe["placer"])
        else:
            placer_type = "pack_col"

        if placer_type not in PLACER_NAME2FUNC:
            raise ValueError(
                f"{placer_type} is not an available placer, Choose:"
                f" {list(PLACER_NAME2FUNC.keys())}"
            )
        _placer = PLACER_NAME2FUNC[placer_type]

        # All other attributes are assumed to be settings for the placer

        ## Check if the cell should be attached to a specific parent cell
        if "parent" in settings:
            parent_name = settings.pop("parent")
            if parent_name not in CELLS:
                # Create parent cell in layout and insert it under top level
                parent_cell = top_level_layout.create_cell(parent_name)
                CELLS[parent_name] = parent_cell
                parent_cell_instance = pya.CellInstArray(
                    parent_cell.cell_index(), transform_identity
                )
                top_level.insert(parent_cell_instance)
            doe_parent_cell = CELLS[parent_name]
        else:
            # If no parent specified, insert the DOE at top level
            doe_parent_cell = top_level

        ## Check if we should create a DOE cell which regroups the DOEs
        if "with_doe_cell" in settings:
            with_doe_cell = settings.pop("with_doe_cell")
        else:
            with_doe_cell = True

        # x0, y0 can either be float or string
        x0 = settings.pop("x0")
        y0 = settings.pop("y0")

        # Check whether we are doing relative or absolute placement
        # if (x0 in ["E", "W"] or y0 in ["N", "S"]) and not placed_doe:
        #     raise ValueError(
        #         "At least one DOE must be placed to use relative placement"
        #     )

        # For relative placement (to previous DOE)
        if "margin_x" not in settings:
            settings["margin_x"] = settings["margin"]
        if "margin_y" not in settings:
            settings["margin_y"] = settings["margin"]

        if "inter_margin_x" not in settings:
            inter_margin_x = settings["margin_x"]
        else:
            inter_margin_x = settings.pop("inter_margin_x")

        if "inter_margin_y" not in settings:
            inter_margin_y = settings["margin_y"]
        else:
            inter_margin_y = settings.pop("inter_margin_y")

        align_x = settings["align_x"]
        align_y = settings["align_y"]

        ## Making sure that the alignment is sensible depending on how we stack

        # If we specify a DOE to place next to, use it
        if "next_to" in settings:
            placed_doe = placed_does[settings.pop("next_to")]

        print(placed_doe)
        print(placed_does)

        # Otherwise, use previously placed DOE as starting point
        doe_si = (
            SizeInfo(placed_doe, top_level_layout, um_to_grid=um_to_grid)
            if placed_doe is not None
            else None
        )
        if x0 == "E":
            x0 = doe_si.east
            if align_x == "W":
                x0 += inter_margin_x

        if x0 == "W":
            x0 = doe_si.west
            if align_x == "E":
                x0 -= inter_margin_x

        if y0 == "N":
            y0 = doe_si.north
            if align_y == "S":
                y0 += inter_margin_y

        if y0 == "S":
            y0 = doe_si.south
            if align_y == "N":
                y0 -= inter_margin_y

        # Add x0, y0 in settings as float
        settings["x0"] = x0
        settings["y0"] = y0

        settings["um_to_grid"] = um_to_grid

        placed_components = _placer(components, **settings)

        # Place components within a cell having the DOE name

        if with_doe_cell or len(placed_components) > 1:
            doe_cell = top_level_layout.create_cell(doe_name)
            CELLS[doe_name] = doe_cell
            for instance in placed_components:
                doe_cell.insert(instance)
            placed_does[doe_name] = doe_cell
            placed_doe = doe_cell
            doe_instance = pya.CellInstArray(doe_cell.cell_index(), transform_identity)
        else:
            # If only single cell and we want to skip the doe cell
            doe_instance = placed_components[0]
            placed_does[doe_name] = doe_instance
            placed_doe = doe_instance

        add_doe_label = doe["add_doe_label"]
        add_doe_visual_label = doe["add_doe_visual_label"]

        if add_doe_label:
            label_layer_index, label_layer_datatype = layer_doe_label
            layer_index = top_level.layout().insert_layer(
                pya.LayerInfo(label_layer_index, label_layer_datatype)
            )
            # Add the name of the DOE at the center of the cell
            _p = doe_instance.bbox(top_level_layout).center()
            _text = pya.Text(doe_name, _p.x, _p.y)
            top_level.shapes(layer_index).insert(_text)

        if add_doe_visual_label:
            _bbox = doe_instance.bbox(top_level_layout)

            idbu = 1 / top_level.layout().dbu
            x_text = _bbox.center().x + doe["dx_visual_label"] * idbu
            y_text = _bbox.bottom + (15.0 + doe["dy_visual_label"]) * idbu
            _text = text.add_text(
                top_level, doe_name, position=(x_text, y_text), fontpath=fontpath
            )
            # _transform = pya.DTrans(x_text, y_text)
            # top_level.insert(pya.CellInstArray(_text.cell_index(), _transform))

        doe_parent_cell.insert(doe_instance)

    return top_level


def place_and_write(
    filepath_yaml, root_does=CONFIG["cache_doe_directory"], filepath_gds="top_level.gds"
):
    c = place_from_yaml(filepath_yaml, root_does)
    _print("writing...")
    c.write(filepath_gds)


def assemble_subdies_from_yaml(filepath, subdies_directory, mask_directory=None):
    data = OmegaConf.load(filepath)
    data = OmegaConf.to_container(data)

    mask = data.pop("mask")
    mask_name = mask["name"]

    # Remaining entries are subdies
    dict_subdies = {
        k: (v["x"], v["y"], v["R"] if "R" in v else 0) for k, v in data.items()
    }

    return assemble_subdies(mask_name, dict_subdies, subdies_directory, mask_directory)


def assemble_subdies(
    mask_name,
    dict_subdies,
    subdies_directory,
    mask_directory=None,
    um_to_grid=UM_TO_GRID,
):
    """
    Args:
        dict_subdies: {subdie_name: (x, y, rotation) in (um, um, deg)}
        subdies_directory: directory where the subdies should be looked for
    """
    top_level_layout = pya.Layout()
    top_level = top_level_layout.create_cell(mask_name)
    if mask_directory is None:
        mask_directory = subdies_directory

    for subdie_name, (x_um, y_um, R) in dict_subdies.items():
        gdspath = os.path.join(subdies_directory, subdie_name + ".gds")
        subdie = load_gds(gdspath).top_cell()

        _subdie = import_cell(top_level_layout, subdie)

        t = pya.Trans(R / 2, 0, int(x_um * um_to_grid), int(y_um * um_to_grid))
        # t = pya.Trans(0, 0)
        subdie_instance = pya.CellInstArray(_subdie.cell_index(), t)
        top_level.insert(subdie_instance)

    top_level.write(os.path.join(mask_directory, mask_name + ".gds"))
    return top_level


def _demo():
    import pp

    c = pp.c.waveguide()
    gdspath = pp.write_component(c)

    layout1 = load_gds(gdspath)
    cell1 = layout1.top_cell()
    cell1_instance1 = pya.CellInstArray(cell1.cell_index(), pya.Trans(10, 0))

    layout2 = pya.Layout()
    layout2.create_cell("TOP_LEVEL")

    layout2.cell("TOP_LEVEL").insert(cell1_instance1)
    layout2.write("test.gds")


if __name__ == "__main__":
    _demo()
    print(CELLS)
