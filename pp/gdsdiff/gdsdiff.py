import itertools
import pathlib
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union

import gdspy as gp
from gdspy.polygon import PolygonSet
from numpy import int64, ndarray

from pp.component import Component
from pp.import_gds import import_gds
from pp.types import ComponentOrReference, Layer

COUNTER = itertools.count()


def boolean(
    A: Iterable[ComponentOrReference],
    B: Iterable[ComponentOrReference],
    operation: str,
    precision: float,
) -> Optional[PolygonSet]:
    p = gp.boolean(
        operand1=A,
        operand2=B,
        operation=operation,
        precision=precision,
        max_points=4000,
    )
    return p


def get_polygons_on_layer(
    cell: Component, layer: Union[Tuple[int64, int64], Tuple[int, int]]
) -> Optional[List[ndarray]]:
    polygons = cell.get_polygons(by_spec=True)
    # pprint ({k: len(v) for k, v in polygons.items()})
    if layer in polygons:
        return polygons[layer]
    else:
        return None


def gdsdiff(
    cellA: Union[Path, Component, str],
    cellB: Union[Path, Component, str],
    name: str = "TOP",
    boolean: bool = False,
) -> Component:
    """Compare two Components.

    Args:
        CellA: Component or path to gds file
        CellB: Component or path to gds file

    Returns:
        Component with both cells (xor, common and diffs)
    """
    if isinstance(cellA, pathlib.Path):
        cellA = str(cellA)
    if isinstance(cellB, pathlib.Path):
        cellB = str(cellB)
    if isinstance(cellA, str):
        cellA = import_gds(cellA, flatten=True)
    if isinstance(cellB, str):
        cellB = import_gds(cellB, flatten=True)

    layers: Set[Layer] = set()
    layers.update(cellA.get_layers())
    layers.update(cellB.get_layers())

    top = Component(name=f"{name}_diffs")

    cellA.name = f"{name}_old"
    cellB.name = f"{name}_new"
    top << cellA
    top << cellB

    if boolean:
        diff = Component(name=f"{name}_xor")
        common = Component(name=f"{name}_common")
        old_only = Component(name=f"{name}_only_in_old")
        new_only = Component(name=f"{name}_only_in_new")

        for layer in layers:
            A = get_polygons_on_layer(cellA, layer)
            B = get_polygons_on_layer(cellB, layer)

            # Common bits
            common_AB = boolean(A, B, operation="and", precision=0.001)

            # Bits in either A or B
            either_AB = boolean(A, B, operation="xor", precision=0.001)

            # Bits only in A
            only_in_A = boolean(A, either_AB, operation="and", precision=0.001)

            # Bits only in B
            only_in_B = boolean(B, either_AB, operation="and", precision=0.001)

            if either_AB is not None:
                diff.add_polygon(either_AB, layer)
            if common_AB is not None:
                common.add_polygon(common_AB, layer)
            if only_in_A is not None:
                old_only.add_polygon(only_in_A, layer)
            if only_in_B is not None:
                new_only.add_polygon(only_in_B, layer)

        top << diff
        top << common
        top << old_only
        top << new_only
    return top


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: gdsdiff <mask_v1.gds> <mask_v2.gds>")
        print("Note that you need to have KLayout opened with klive running")
        sys.exit()

    c = gdsdiff(sys.argv[1], sys.argv[2])
    c.show()
