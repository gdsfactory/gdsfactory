import phidl.geometry as pg

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def xor_diff(A, B, precision: float = 1e-4) -> Component:
    """Given two Devices A and B, performs the layer-by-layer XOR
    difference between A and B and returns polygons representing the
    differences between A and B.

    gdsfactory wrapper for phidl.geometry.xor_diff

    Args:
        A: Component(/Reference) or list of Component(/References)
        B: Component(/Reference) or list of Component(/References)
        precision: Desired precision for rounding vertex coordinates.

    Returns
        Component: containing a polygon(s) defined by the XOR difference result
        between A and B.
    """
    return gf.read.from_phidl(component=pg.xor_diff(A, B, precision=precision))


if __name__ == "__main__":
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = xor_diff(A=e1, B=e2)
    c.show()
