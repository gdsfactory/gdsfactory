import sys

import gdspy as gp

from pp.geo_utils import area


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def compute_area(c, target_layer):
    """
    Compute area of the component on a given layer
    """
    _print("Computing area ", c.name)
    c.flatten()
    # return c.area(by_spec=True)[layer]
    polys_by_spec = c.get_polygons(by_spec=True)
    _area = 0
    for (layer, polys) in polys_by_spec.items():
        _print(layer)
        if layer == target_layer:
            joined_polys = gp.boolean(polys, None, operation="or")
            _print(joined_polys)
            try:
                _area += sum([abs(area(p)) for p in joined_polys.polygons])
            except BaseException:
                print(f"Warning, {c.name} joinedpoly {joined_polys} could not be added")
    return _area


def bucket_cells_by_rank(cells):
    cells = list(cells)
    rank = 0
    rank_to_cells = {}
    all_classified_cells = set()
    prev_len_cells = -1
    while cells:
        classified_cells = set()
        to_rm = []
        for i, c in enumerate(cells):
            _cells = c.get_dependencies(recursive=False)
            unclassified_subcells = _cells - all_classified_cells
            if len(unclassified_subcells) == 0:
                classified_cells.update([c])
                to_rm += [i]

        if prev_len_cells == len(cells):
            print(cells)
            raise ValueError("Error: some cells cannot be linked")
        prev_len_cells = len(cells)
        while to_rm:
            cells.pop(to_rm.pop())

        rank_to_cells[rank] = classified_cells
        all_classified_cells.update(classified_cells)
        rank += 1
    return rank_to_cells


def get_polygons_on_layer(c, layer):
    polygons = []
    for polyset in c.polygons:
        for ii in range(len(polyset.polygons)):
            key = (polyset.layers[ii], polyset.datatypes[ii])
            if key == layer:
                polygons.append(polyset.polygons[ii])
    return polygons


def boolops_hierarchical(
    c, layer1, layer2, layer_result, operation="or", func_check_to_flatten=None
):

    all_cells = c.get_dependencies(recursive=True)
    all_cells.update([c])
    cells_by_rank = bucket_cells_by_rank(all_cells)
    _print("Found the hierarchy...")

    if func_check_to_flatten is None:

        def _has_polygons(cell):
            n = 0
            for _layer in [layer1, layer2]:
                polys = get_polygons_on_layer(cell, _layer)
                n += len(polys)
            return n

        func_check_to_flatten = _has_polygons

    for rank, cells in cells_by_rank.items():
        for cell in cells:
            to_flatten = func_check_to_flatten(cell)
            if to_flatten:
                _print("BOOL HIERARCHY", cell.name, "...", end="")
                polys_by_spec = cell.get_polygons(by_spec=True)
                polys1 = polys_by_spec[layer1] if layer1 in polys_by_spec else []
                polys2 = polys_by_spec[layer2] if layer2 in polys_by_spec else []
                cell.remove_layers([layer_result])

                res_polys = gp.boolean(polys1, polys2, operation=operation)
                if res_polys is not None:
                    cell.add_polygon(res_polys, layer=layer_result)

                _print("{} - done".format(res_polys), cell.name)

    return cell


def compute_area_hierarchical(
    c, layer, func_check_to_flatten=None, keep_zero_area_cells=False
):
    """
    Compute area of the component on a given layer
    Faster than `compute_area` but need to be careful if the cells overlap
    Can pass a list of cells to flatten
    """

    all_cells = c.get_dependencies(recursive=True)
    all_cells.update([c])
    cells_by_rank = bucket_cells_by_rank(all_cells)
    _print("Found the hierarchy...")

    cell_to_area = {}
    cell_to_rank = {}

    if func_check_to_flatten is None:

        def _has_polygons(cell):
            polys = get_polygons_on_layer(cell, layer)
            return len(polys)

        func_check_to_flatten = _has_polygons

    for rank, cells in cells_by_rank.items():
        for cell in cells:
            to_flatten = func_check_to_flatten(cell)
            if to_flatten:
                _print("CAH - TO FLATTEN", to_flatten)
                _area = compute_area(cell, layer)
            else:
                # _cell_area_by_spec = cell.area(by_spec=True)
                _area = 0
                # _cell_area_by_spec[layer] if layer in _cell_area_by_spec else 0
                # _area = 0

                # _print("CAH - ",cell.name)

                for ref in cell.references:
                    _area += cell_to_area[ref.ref_cell.name]
                _print(
                    "CAH {} {:.1f} {}".format(cell.name, _area, len(cell.references))
                )

            cell_to_area[cell.name] = _area
            cell_to_rank[cell.name] = rank
    to_rm = []
    if not keep_zero_area_cells:
        for k, v in cell_to_area.items():
            if v == 0:
                to_rm += [k]

    while to_rm:
        cell_to_area.pop(to_rm.pop())

    # add rank
    cell_to_data = {}
    for k, v in cell_to_area.items():
        cell_to_data[k] = (v, cell_to_rank[k])

    return cell_to_data


if __name__ == "__main__":
    import pp

    c = pp.c.mzi2x2()
    print(bucket_cells_by_rank([c] + list(c.get_dependencies(recursive=True))))
