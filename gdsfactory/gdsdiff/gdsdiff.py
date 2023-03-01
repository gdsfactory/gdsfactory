from __future__ import annotations

import itertools
import pathlib
from pathlib import Path
from typing import Union

import gdstk

from gdsfactory.component import Component
from gdsfactory.read.import_gds import import_gds

COUNTER = itertools.count()


def xor_polygons(A: Component, B: Component, hash_geometry: bool = True):
    """Given two devices A and B, performs a layer-by-layer XOR diff between A \
    and B, and returns polygons representing the differences between A and B.

    Adapted from lytest/kdb_xor.py

    """
    # first do a geometry hash to vastly speed up if they are equal
    if hash_geometry and (A.hash_geometry() == B.hash_geometry()):
        return Component()

    D = Component()
    A_polys = A.get_polygons(by_spec=True)
    B_polys = B.get_polygons(by_spec=True)
    A_layers = A_polys.keys()
    B_layers = B_polys.keys()
    all_layers = set()
    all_layers.update(A_layers)
    all_layers.update(B_layers)
    for layer in all_layers:
        if (layer in A_layers) and (layer in B_layers):
            p = gdstk.boolean(
                A_polys[layer],
                B_polys[layer],
                operation="xor",
                precision=0.001,
                layer=layer[0],
                datatype=layer[1],
            )
        elif layer in A_layers:
            p = A_polys[layer]
        elif layer in B_layers:
            p = B_polys[layer]
        if p is not None:
            for polygon in p:
                D.add_polygon(polygon, layer=layer)
    return D


def gdsdiff(
    component1: Union[Path, Component, str],
    component2: Union[Path, Component, str],
    name: str = "TOP",
    xor: bool = True,
) -> Component:
    """Returns two Components overlay and diffs (XOR).

    Args:
        component1: Component or path to gds file (reference).
        component2: Component or path to gds file (run).
        name: name of the top cell.
        xor: if True includes boolean operation.

    """
    if isinstance(component1, (str, pathlib.Path)):
        component1 = import_gds(str(component1), flatten=True, name=f"{name}_old")
    if isinstance(component2, (str, pathlib.Path)):
        component2 = import_gds(str(component2), flatten=True, name=f"{name}_new")

    component1 = component1.copy()
    component2 = component2.copy()

    component1.name = f"{name}_old"
    component2.name = f"{name}_new"

    top = Component(name=f"{name}_diffs")
    ref1 = top << component1
    ref2 = top << component2

    if xor:
        diff = xor_polygons(ref1, ref2, hash_geometry=False)
        diff.name = f"{name}_xor"
        top.add_ref(diff)

    return top


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: gdsdiff <mask_v1.gds> <mask_v2.gds>")
        print("Note that you need to have KLayout opened with klive running")
        sys.exit()
    c = gdsdiff(sys.argv[1], sys.argv[2])
    c.show(show_ports=True)
