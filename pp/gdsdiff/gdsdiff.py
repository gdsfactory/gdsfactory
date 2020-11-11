import itertools
import pathlib
import gdspy as gp

from pp import import_gds
import pp

COUNTER = itertools.count()


def boolean(A, B, operation, precision):
    p = gp.boolean(
        operand1=A,
        operand2=B,
        operation=operation,
        precision=precision,
        max_points=4000,
    )
    return p


def get_polygons_on_layer(cell, layer):
    polygons = cell.get_polygons(by_spec=True)
    # pprint ({k: len(v) for k, v in polygons.items()})
    if layer in polygons:
        return polygons[layer]
    else:
        return None


def gdsdiff(cellA, cellB):
    """
    Args:
        CellA: gds cell (as pp.Component) or path to gds file
        CellB: gds cell (as pp.Component) or path to gds file

    Output:
        gds file containing the diff between the two GDS files
    """
    if isinstance(cellA, pathlib.PosixPath):
        cellA = str(cellA)
    if isinstance(cellB, pathlib.PosixPath):
        cellB = str(cellB)
    if type(cellA) == str:
        cellA = import_gds(cellA, flatten=True)
    if type(cellB) == str:
        cellB = import_gds(cellB, flatten=True)

    layers = set()
    layers.update(cellA.get_layers())
    layers.update(cellB.get_layers())

    diff = pp.Component(name="diff")
    for layer in layers:
        # We go to "process" layer beyond 1000 to put the diff
        diff_process = layer[0] + 1000

        """
        # datatype is used to differentiate the diff
        # 0: unchanged
        # 1: removed (assuming B is the updated version of A)
        # 2: added (assuming B is the updated version of A)
        """

        layer_common = (diff_process, 0)
        layer_only_A = (diff_process, 1)
        layer_only_B = (diff_process, 2)

        A = get_polygons_on_layer(cellA, layer)
        B = get_polygons_on_layer(cellB, layer)

        if A is None and B is None:
            continue
        elif B is None:
            diff.add_polygon(A, layer_only_A)
            continue
        elif A is None:
            diff.add_polygon(B, layer_only_B)
            continue

        # Common bits
        common_AB = boolean(A, B, operation="and", precision=0.001)

        # Bits in either A or B
        either_AB = boolean(A, B, operation="xor", precision=0.001)

        # Bits only in A
        only_in_A = boolean(A, either_AB, operation="and", precision=0.001)

        # Bits only in B
        only_in_B = boolean(B, either_AB, operation="and", precision=0.001)

        if common_AB is not None:
            diff.add_polygon(common_AB, layer_common)
        if only_in_A is not None:
            diff.add_polygon(only_in_A, layer_only_A)
        if only_in_B is not None:
            diff.add_polygon(only_in_B, layer_only_B)

    return diff


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: gdsdiff <mask_v1.gds> <mask_v2.gds>")
        print("Note that you need to have KLayout opened with klive running")
        sys.exit()

    diff = gdsdiff(sys.argv[1], sys.argv[2])
    pp.show(diff)
