from typing import Tuple, Union

import phidl.geometry as pg

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import ComponentOrReference, Int2, Layer


@gf.cell
def boolean(
    A: Union[ComponentOrReference, Tuple[ComponentOrReference, ...]],
    B: Union[ComponentOrReference, Tuple[ComponentOrReference, ...]],
    operation: str,
    precision: float = 1e-4,
    num_divisions: Union[int, Int2] = (1, 1),
    max_points: int = 4000,
    layer: Layer = (1, 0),
) -> Component:
    """Performs boolean operations between 2 Component/Reference objects,
    or lists of Devices/DeviceReferences.

    ``operation`` should be one of {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
    Note that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched

    gdsfactory wrapper for phidl.geometry.boolean

    You can also use gdsfactory.drc.boolean that uses Klayout backend

    Args:
        A: Component(/Reference) or list of Component(/References)
        B: Component(/Reference) or list of Component(/References)
        operation: {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
        precision: float Desired precision for rounding vertex coordinates.
        num_divisions: number of divisions with which the geometry is divided into
          multiple rectangular regions. This allows for each region to be
          processed sequentially, which is more computationally efficient.
        max_points: The maximum number of vertices within the resulting polygon.
        layer: Specific layer to put polygon geometry on.

    Returns: Component with polygon(s) of the boolean operations between
      the 2 input Devices performed.

    Notes
    -----
    'A+B' is equivalent to 'or'.
    'A-B' is equivalent to 'not'.
    'B-A' is equivalent to 'not' with the operands switched.
    """

    A = list(A) if isinstance(A, tuple) else A
    B = list(B) if isinstance(B, tuple) else B

    c = pg.boolean(
        A=A,
        B=B,
        operation=operation,
        precision=precision,
        num_divisions=num_divisions,
        max_points=max_points,
        layer=layer,
    )
    return gf.read.from_phidl(component=c)


def test_boolean():
    c = gf.Component()
    e1 = c << gf.components.ellipse()
    e2 = c << gf.components.ellipse(radii=(10, 6))
    e3 = c << gf.components.ellipse(radii=(10, 4))
    e3.movex(5)
    e2.movex(2)
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    assert len(c.polygons) == 2, len(c.polygons)


if __name__ == "__main__":
    c = gf.Component()
    e1 = c << gf.components.ellipse()
    e2 = c << gf.components.ellipse(radii=(10, 6))
    e3 = c << gf.components.ellipse(radii=(10, 4))
    e3.movex(5)
    e2.movex(2)
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    c.show()
