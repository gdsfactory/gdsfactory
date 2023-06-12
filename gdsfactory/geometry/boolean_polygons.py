from __future__ import annotations

from typing import List, Union

import gdstk

import gdsfactory as gf
from gdsfactory.typings import Component, ComponentReference, LayerSpec


def boolean_polygons(
    operand1: Union[ComponentReference, Component, gdstk.Polygon],
    operand2: Union[ComponentReference, Component, gdstk.Polygon],
    operation: str,
    output_layer: LayerSpec = (0, 0),
    precision: float = 1e-3,
) -> List[gdstk.Polygon]:
    """Perform a boolean operation and return the list of resulting Polygons.
    See [gdstk docs](https://heitzmann.github.io/gdstk/geometry/gdstk.boolean.html#gdstk.boolean) for details.

    Args:
        operand1: polygon set A.
        operand2: polygon set B.
        operation: the name of the operation to perform, i.e. "or", "and", "not", or "xor".
        output_layer: the layer to assign the resulting polygons.
        precision: the precision used for the operation, in microns.

    Returns: a list of gdstk Polygons on the specified output layer.
    """
    layer, datatype = gf.get_layer(output_layer)

    if operand2 is None:
        operand2 = []
    if hasattr(operand1, "get_polygons"):
        operand1 = operand1.get_polygons(as_array=False)
    if hasattr(operand2, "get_polygons"):
        operand2 = operand2.get_polygons(as_array=False)

    return gdstk.boolean(
        operand1=operand1,
        operand2=operand2,
        operation=operation,
        precision=precision,
        layer=layer,
        datatype=datatype,
    )


if __name__ == "__main__":
    import time

    n = 50
    c1 = gf.c.array(gf.c.circle(radius=10), columns=n, rows=n)
    c2 = gf.c.array(gf.c.circle(radius=9), columns=n, rows=n).ref()
    c2.movex(5)

    t0 = time.time()
    p = boolean_polygons(c1, c2, operation="xor")

    t1 = time.time()
    print(t1 - t0)

    c = gf.Component("demo")
    c.add_polygon(p)
    c.show(show_ports=True)
