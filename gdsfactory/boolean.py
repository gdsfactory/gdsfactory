from typing import List, Optional, Union

import phidl.geometry as pg

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import ComponentOrReference, Layer


def boolean(
    A: Union[ComponentOrReference, List[ComponentOrReference]],
    B: Union[ComponentOrReference, List[ComponentOrReference]],
    operation: str,
    precision: float = 1e-4,
    num_divisions: Optional[int] = None,
    max_points: int = 4000,
    layer: Layer = (1, 0),
) -> Component:
    """Performs boolean operations between 2 Device/DeviceReference objects,
    or lists of Devices/DeviceReferences.

    ``operation`` should be one of {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
    Note that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched

    Args:
        A : Device(/Reference) or list of Device(/Reference) or Polygon Input Devices.
        B : Device(/Reference) or list of Component(/Reference) or Polygon Input Devices.
        operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'} Boolean operation to perform.
        precision : float Desired precision for rounding vertex coordinates.
        num_divisions : array-like[2] of int
            number of divisions with which the geometry is divided into
            multiple rectangular regions. This allows for each region to be
            processed sequentially, which is more computationally efficient.
        max_points :
            The maximum number of vertices within the resulting polygon.
        layer : int, array-like[2], or set
            Specific layer(s) to put polygon geometry on.

    Returns:  Device
        A Component containing a polygon(s) with the boolean operations between
        the 2 input Devices performed.

    Notes
    -----
    'A+B' is equivalent to 'or'.
    'A-B' is equivalent to 'not'.
    'B-A' is equivalent to 'not' with the operands switched.
    """
    num_divisions = num_divisions or [1, 1]
    c = pg.boolean(
        A=A,
        B=B,
        operation=operation,
        precision=precision,
        num_divisions=num_divisions,
        max_points=max_points,
        layer=layer,
    )
    return gf.read.phidl(component=c)


if __name__ == "__main__":
    e1 = gf.components.ellipse()
    e2 = gf.components.ellipse(radii=(10, 6)).movex(2)
    e3 = gf.components.ellipse(radii=(10, 4)).movex(5)
    # gf.plot([e1, e2, e3])
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    c.show()
