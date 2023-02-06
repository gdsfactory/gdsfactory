from __future__ import annotations

import gdstk

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Layer


def _union_polygons(polygons, precision: float = 1e-4):
    """Performs union of polygons within PolygonSet or list of polygons.

    Args:
        polygons : PolygonSet or list of polygons
            A set containing the input polygons.
        precision : float
            Desired precision for rounding vertex coordinates.
        max_points : int
            The maximum number of vertices within the resulting polygon.

    Returns
        unioned: polygon The result of the union of all the polygons
            within the input PolygonSet.
    """
    return gdstk.boolean(
        polygons,
        [],
        operation="or",
        precision=precision,
    )


@gf.cell
def union(
    component: Component,
    by_layer: bool = False,
    precision: float = 1e-4,
    join_first: bool = True,
    layer: Layer = (1, 0),
) -> Component:
    """Returns inverted union of Component polygons.

    based on phidl.geometry.invert

    Args:
        component: Component(/Reference), list of Component(/Reference), or Polygon
            A containing the polygons to perform union and inversion on.
        by_Layer: performs the union operation layer-wise so each layer can be
            individually combined.
        precision: Desired precision for rounding vertex coordinates.
        join_first: before offsetting to avoid unnecessary joins
            in adjacent polygons.
        layer: Specific layer to put polygon geometry on.

    """
    U = Component()

    if by_layer:
        all_polygons = component.get_polygons(by_spec=True)
        for layer, polygons in all_polygons.items():
            unioned_polygons = _union_polygons(
                polygons,
                precision=precision,
            )
            U.add_polygon(unioned_polygons, layer=layer)
    else:
        all_polygons = component.get_polygons(by_spec=False)
        unioned_polygons = _union_polygons(
            all_polygons,
            precision=precision,
        )
        U.add_polygon(unioned_polygons, layer=layer)
    return U


def test_union() -> None:
    c = Component()
    c << gf.components.ellipse(radii=(6, 6)).move((12, 10))
    c << gf.components.ellipse(radii=(10, 4))
    c2 = union(c)
    assert int(c2.area()) == 238, c2.area()


if __name__ == "__main__":
    test_union()
    c = Component()
    c << gf.components.ellipse(radii=(6, 6)).move((12, 10))
    c << gf.components.ellipse(radii=(10, 4))
    c2 = union(c)
    c2.show()
