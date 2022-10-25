import gdspy as gp
from numpy import float64

from gdsfactory.component import Component
from gdsfactory.geometry.functions import area
from gdsfactory.types import Dict, Layer, Tuple


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
            # print(cells)
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


def compute_area(component: Component, layer: Layer) -> float64:
    """Returns Computed area of the component for a given layer."""
    c = component.copy()
    c.flatten()
    polys_by_spec = c.get_polygons(by_spec=True)
    _area = 0
    for layer_polygons, polys in polys_by_spec.items():
        # print(layer)
        if layer_polygons == layer:
            joined_polys = gp.boolean(polys, None, operation="or")
            # print(joined_polys)
            try:
                _area += sum(abs(area(p)) for p in joined_polys.polygons)
            except BaseException:
                print(f"Warning, {c.name} joinedpoly {joined_polys} could not be added")
    return _area


def compute_area_hierarchical(
    component: Component,
    layer: Layer,
    func_check_to_flatten=None,
    keep_zero_area_cells: bool = False,
) -> Dict[str, Tuple[float, int]]:
    """Compute area of the component on a given layer Faster than \
    `compute_area` but need to be careful if the cells overlap Can pass a list \
    of cells to flatten Returns Dict[key of each cell, Tuple[area, rank \
    (position in hierarchy)].

    Args:
        component:
        layer:
        func_check_to_flatten:
        keep_zero_area_cells:removes zero area cells

    """
    all_cells = component.get_dependencies(recursive=True)
    all_cells.update([component])
    cells_by_rank = bucket_cells_by_rank(all_cells)
    # print("Found the hierarchy...")

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
                # print("CAH - TO FLATTEN", to_flatten)
                _area = compute_area(cell, layer)
            else:
                # _cell_area_by_spec = cell.area(by_spec=True)
                _area = 0
                # _cell_area_by_spec[layer] if layer in _cell_area_by_spec else 0
                # _area = 0

                # print("CAH - ",cell.name)

                for ref in cell.references:
                    _area += cell_to_area[ref.ref_cell.name]
                # print(
                #     "CAH {} {:.1f} {}".format(cell.name, _area, len(cell.references))
                # )

            cell_to_area[cell.name] = _area
            cell_to_rank[cell.name] = rank
    to_rm = []
    if not keep_zero_area_cells:
        for k, v in cell_to_area.items():
            if v == 0:
                to_rm += [k]

    while to_rm:
        cell_to_area.pop(to_rm.pop())

    cell_to_data = {}
    for k, v in cell_to_area.items():
        cell_to_data[k] = (v, cell_to_rank[k])

    return cell_to_data


def test_compute_area() -> None:
    import gdsfactory as gf

    c = gf.components.mzi()
    assert int(compute_area(c, layer=(1, 0))) == 148, int(compute_area(c, layer=(1, 0)))


def test_compute_area_hierarchical() -> None:
    import gdsfactory as gf

    c = gf.components.mzi()
    assert int(compute_area_hierarchical(c, layer=(1, 0))[c.name][0]) == 148, int(
        compute_area_hierarchical(c, layer=(1, 0))[c.name][0]
    )


if __name__ == "__main__":
    test_compute_area_hierarchical()
    # test_compute_area()
    # import gdsfactory as gf
    # print(bucket_cells_by_rank([c] + list(c.get_dependencies(recursive=True))))
    # c = gf.components.mzi()
    # print(compute_area(c, layer=(1, 0)))
    # d = compute_area_hierarchical(c, layer=(1, 0))
    # c.show(show_ports=True)
    # test_compute_area_hierarchical()
    # test_compute_area()
