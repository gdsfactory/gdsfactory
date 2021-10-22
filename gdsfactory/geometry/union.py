import phidl.geometry as pg

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import Layer


@gf.cell
def union(
    component: Component,
    by_layer: bool = False,
    precision: float = 1e-4,
    join_first: bool = True,
    max_points: int = 4000,
    layer: Layer = (1, 0),
) -> Component:
    """Creates an inverted version of the input shapes with an additional
    border around the edges.

    adapted from phidl.geometry.invert

    Args:
        D: Component(/Reference), list of Component(/Reference), or Polygon
            A Component containing the polygons to perform union on.
        by_Layer: performs the union operation layer-wise so each layer can be
            individually combined.
        precision: Desired precision for rounding vertex coordinates.
        join_first: before offsetting to avoid unnecessary joins in adjacent polygon
        max_points: The maximum number of vertices within the resulting polygon.
        layer : Specific layer to put polygon geometry on.

    Returns
        Component containing the  union of the polygons
    """
    U = Component()

    if by_layer:
        all_polygons = component.get_polygons(by_spec=True)
        for layer, polygons in all_polygons.items():
            unioned_polygons = pg._union_polygons(
                polygons, precision=precision, max_points=max_points
            )
            U.add_polygon(unioned_polygons, layer=layer)
    else:
        all_polygons = component.get_polygons(by_spec=False)
        unioned_polygons = pg._union_polygons(
            all_polygons, precision=precision, max_points=max_points
        )
        U.add_polygon(unioned_polygons, layer=layer)
    return U


def test_union():
    c = Component()
    c << gf.components.ellipse(radii=(6, 6)).move((12, 10))
    c << gf.components.ellipse(radii=(10, 4))
    c2 = union(c)
    assert int(c2.area()) == 238, c2.area()


if __name__ == "__main__":
    c = Component()
    c << gf.components.ellipse(radii=(6, 6)).move((12, 10))
    c << gf.components.ellipse(radii=(10, 4))
    c2 = union(c)
    c2.show()
