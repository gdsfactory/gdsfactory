import phidl.geometry as pg

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def outline(elements, **kwargs) -> Component:
    """
    Returns Component containing the outlined polygon(s).

    wraps phidl.geometry.outline

    Creates an outline around all the polygons passed in the `elements`
    argument. `elements` may be a Device, Polygon, or list of Devices.

    Args:
        elements: Device(/Reference), list of Device(/Reference), or Polygon
            Polygons to outline or Device containing polygons to outline.

    Keyword Args:
        distance: int or float
            Distance to offset polygons. Positive values expand, negative shrink.
        precision: float
            Desired precision for rounding vertex coordinates.
        num_divisions: array-like[2] of int
            The number of divisions with which the geometry is divided into
            multiple rectangular regions. This allows for each region to be
            processed sequentially, which is more computationally efficient.
        join: {'miter', 'bevel', 'round'}
            Type of join used to create the offset polygon.
        tolerance: int or float
            For miter joints, this number must be at least 2 and it represents the
            maximal distance in multiples of offset between new vertices and their
            original position before beveling to avoid spikes at acute joints. For
            round joints, it indicates the curvature resolution in number of
            points per full circle.
        join_first: bool
            Join all paths before offsetting to avoid unnecessary joins in
            adjacent polygon sides.
        max_points: int
            The maximum number of vertices within the resulting polygon.
        open_ports: bool or float
            If not False, holes will be cut in the outline such that the Ports are
            not covered. If True, the holes will have the same width as the Ports.
            If a float, the holes will be be widened by that value (useful for fully
            clearing the outline around the Ports for positive-tone processes
        layer: int, array-like[2], or set
            Specific layer(s) to put polygon geometry on.)

    """
    return gf.read.from_phidl(component=pg.outline(elements, **kwargs))


def test_outline():
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = outline([e1, e2])
    assert int(c.area()) == 52


if __name__ == "__main__":
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = outline([e1, e2])
    c.show()
