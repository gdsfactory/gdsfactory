import itertools
import pathlib
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union

import gdspy as gp
from gdspy.polygon import PolygonSet
from numpy import int64, ndarray

from gdsfactory.component import Component
from gdsfactory.import_gds import import_gds
from gdsfactory.types import ComponentOrReference, Layer

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
    component1: Union[Path, Component, str],
    component2: Union[Path, Component, str],
    name: str = "TOP",
    make_boolean: bool = False,
) -> Component:
    """Compare two Components.

    Args:
        component1: Component or path to gds file
        component2: Component or path to gds file
        name: name of the top cell
        make_boolean: makes boolean operation

    Returns:
        Component with both cells (xor, common and diffs)
    """
    if isinstance(component1, pathlib.Path):
        component1 = str(component1)
    if isinstance(component2, pathlib.Path):
        component2 = str(component2)
    if isinstance(component1, str):
        component1 = import_gds(component1, flatten=True)
    if isinstance(component2, str):
        component2 = import_gds(component2, flatten=True)

    layers: Set[Layer] = set()
    layers.update(component1.get_layers())
    layers.update(component2.get_layers())

    top = Component(name=f"{name}_diffs")

    if component1.name.startswith("Unnamed"):
        component1.name = f"{name}_old"
    if component2.name.startswith("Unnamed"):
        component2.name = f"{name}_new"

    top << component1
    top << component2

    if make_boolean:
        diff = Component(name=f"{name}_xor")
        common = Component(name=f"{name}_common")
        old_only = Component(name=f"{name}_only_in_old")
        new_only = Component(name=f"{name}_only_in_new")

        for layer in layers:
            A = get_polygons_on_layer(component1, layer)
            B = get_polygons_on_layer(component2, layer)

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
    # import sys

    # if len(sys.argv) != 3:
    #     print("Usage: gdsdiff <mask_v1.gds> <mask_v2.gds>")
    #     print("Note that you need to have KLayout opened with klive running")
    #     sys.exit()

    # c = gdsdiff(sys.argv[1], sys.argv[2])
    # c.show()

    import gdsfactory as gf

    c = gdsdiff(gf.components.straight(), gf.components.straight(length=11))
    c.show()
