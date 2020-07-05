"""
From two GDS files, find which cells are identical
and which cells with the same name are different
This is not a diff tool
"""
import hashlib
import sys
import json
import numpy as np


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def get_transform(cell_ref, precision=1e-4):
    """
    Get the transform from a cell-instance as a hashable object
    """
    if cell_ref.rotation is None:
        cell_ref.rotation = 0
    return (
        int(cell_ref.origin[0] / precision),
        int(cell_ref.origin[1] / precision),
        int(cell_ref.rotation) % 360,
        cell_ref.x_reflection,
    )


def get_polygons_by_spec(cell):
    d = {}
    for _pset in cell.polygons:
        for poly, layer, datatype in zip(_pset.polygons, _pset.layers, _pset.datatypes):
            key = (layer, datatype)
            if key not in d:
                d[key] = []
            d[key].append(poly)

    return d


def get_dependencies_names(cell):
    return [_c.ref_cell.name for _c in cell.references]


def normalize_polygon_start_point(p, dbg=False):
    args_min_x = np.where(p[:, 0] == p[:, 0].min())[0]
    if len(args_min_x) == 1:
        i0 = args_min_x[0]
    else:
        ys = np.array([p[i, 1] for i in args_min_x])
        i0 = args_min_x[np.argmin(ys)]

    if dbg:
        _print(p)
        _print(i0, args_min_x)
        _print()

    return np.roll(p, -i0, axis=0)


def hash_cells(cell, dict_hashes={}, precision=1e-4, dbg_indent=0, dbg=False):
    """
    Algorithm:
    For each polygon directly within this cell:
         - sort the layers

        For each layer, each polygon is individually hashed and then
          the polygon hashes are sorted, to ensure the hash stays constant
          regardless of the ordering the polygons.  Similarly, the layers
          are sorted by (layer, datatype)

    For each cell instance:
        recursively hash the ref_cell + transform
        sort all the hashes for the hash to stay constant regardless of cell instance order

    """
    if cell.name in dict_hashes:
        return dict_hashes

    polygons_by_spec = get_polygons_by_spec(cell)
    layers = list(polygons_by_spec.keys())
    layers.sort()

    if dbg:
        _print(layers)

    """
    # A random offset which fixes common rounding errors intrinsic
    # to floating point math. Example: with a precision of 0.1, the
    # floating points 7.049999 and 7.050001 round to different values
    # (7.0 and 7.1), but offset values (7.220485 and 7.220487) don't
    """
    magic_offset = 0.17048614

    final_hash = hashlib.sha1()

    for layer in layers:
        layer_hash = hashlib.sha1(str(layer).encode()).hexdigest()
        polygons = polygons_by_spec[tuple(layer)]

        polygons = [((p / precision) + magic_offset).astype(np.int64) for p in polygons]
        polygons = [normalize_polygon_start_point(p) for p in polygons]
        polygon_hashes = np.sort([hashlib.sha1(p).hexdigest() for p in polygons])

        if dbg:
            _print(polygons)
            _print(layer, layer_hash, polygon_hashes)

        final_hash.update(layer_hash.encode())
        for ph in polygon_hashes:
            final_hash.update(ph.encode())

    # Ref cell hashes
    cell_ref_uids = []
    for cell_ref in cell.references:
        _cell = cell_ref.ref_cell
        hash_cells(
            _cell,
            precision=precision,
            dict_hashes=dict_hashes,
            dbg_indent=dbg_indent + 2,
            dbg=dbg,
        )
        cell_hash = dict_hashes[_cell.name]
        tr_str = "x{}y{}R{}H{}".format(*get_transform(cell_ref, precision))

        cell_ref_uid = cell_hash + "_" + tr_str
        cell_ref_uids += [cell_ref_uid]

    # Sort hashes (for constant hash regardless of cell_ref ordering)
    cell_ref_uids = np.sort(cell_ref_uids)
    for _hash in cell_ref_uids:
        final_hash.update(_hash.encode())

    dict_hashes[cell.name] = final_hash.hexdigest()
    if dbg:
        _print()
    return dict_hashes


def check_lib_consistency(lib_cells):
    """
    TODO: check that the library does not have a name collision (with different hashes)
    """


def is_leaf_cell(cell):
    return len(cell.references) == 0


def get_cell_status(cell, name_to_hash_teg, name_to_hash_lib, cells_status={}):
    if cell.name in cells_status:
        return cells_status[cell.name]

    sub_cells_names = []
    if cell.name in name_to_hash_lib:
        """
        Known leaf cell: it ether matches the lib or not
        """

        # _print("Known",cell.name , name_to_hash_teg[cell.name][:8], name_to_hash_lib[cell.name][:8])

        if name_to_hash_teg[cell.name] == name_to_hash_lib[cell.name]:
            cell_status = 0
        else:
            cell_status = 2

    elif is_leaf_cell(cell):
        """
        Unknown leaf cell
        """
        cell_status = 1

    else:
        """
        Compound cell not in lib: inherit status from subcells
        """
        sub_cells = [_c.ref_cell for _c in cell.references]
        cell_status = max(
            [
                get_cell_status(_c, name_to_hash_teg, name_to_hash_lib, cells_status)[0]
                for _c in sub_cells
            ]
        )
        sub_cells_names = [_c.name for _c in sub_cells]

    cell_status_and_dependencies = (cell_status, sub_cells_names)
    cells_status[cell.name] = cell_status_and_dependencies
    return cell_status_and_dependencies


def compare_gds_to_lib(teg, lib_cells=[]):
    """
    teg: a cell
    lib_cell: list of cells

    output:
        dictionnary {cell_name: status}
        where status is:
            0 if the cell matches the library, or is composed of cells with status 0
            1 if the cell does not match any lib cell of at least one cell with status 1 and all the others with status 0
            2 if the cell name is in the library but the hashes do not match, OR if the cell is composed of at least one cell with status 2
    """

    name_to_hash_teg = {}
    hash_cells(teg, name_to_hash_teg)
    ds_lib = [hash_cells(_c, {}) for _c in lib_cells]

    check_lib_consistency(ds_lib)
    name_to_hash_lib = {}
    while ds_lib:
        name_to_hash_lib.update(ds_lib.pop())

    with open("dbg.json", "w") as fw:
        fw.write(
            json.dumps({"LIB": name_to_hash_lib, "TEG": name_to_hash_teg}, indent=2)
        )

    # hash_to_name_teg = {v: k for k, v in name_to_hash_teg.items()}
    # hash_to_name_lib = {v: k for k, v in name_to_hash_lib.items()}

    cells_status = {}
    get_cell_status(teg, name_to_hash_teg, name_to_hash_lib, cells_status)
    return cells_status


if __name__ == "__main__":
    pass
