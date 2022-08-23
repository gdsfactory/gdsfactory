import gdspy

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def xor_diff(A, B, precision: float = 1e-4) -> Component:
    """Given two Devices A and B, performs the layer-by-layer XOR difference \
    between A and B and returns polygons representing the differences between A \
    and B.

    gdsfactory wrapper for phidl.geometry.xor_diff

    Args:
        A: Component(/Reference) or list of Component(/References).
        B: Component(/Reference) or list of Component(/References).
        precision: Desired precision for rounding vertex coordinates.

    Returns
        Component: containing a polygon(s) defined by the XOR difference result
        between A and B.

    """
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
            p = gdspy.boolean(
                operand1=A_polys[layer],
                operand2=B_polys[layer],
                operation="xor",
                precision=precision,
                max_points=4000,
                layer=layer[0],
                datatype=layer[1],
            )
        elif layer in A_layers:
            p = A_polys[layer]
        elif layer in B_layers:
            p = B_polys[layer]
        if p is not None:
            D.add_polygon(p, layer=layer)
    return D


if __name__ == "__main__":
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = xor_diff(A=e1, B=e2)
    c.show(show_ports=True)
