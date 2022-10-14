import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.geometry.boolean import boolean
from gdsfactory.geometry.offset import offset


@gf.cell
def outline(
    elements,
    distance=1,
    precision=1e-4,
    num_divisions=(1, 1),
    join="miter",
    tolerance=2,
    join_first=True,
    max_points=4000,
    open_ports=False,
    layer=0,
) -> Component:
    """Returns Component containing the outlined polygon(s).

    Creates an outline around all the polygons passed in the `elements`
    argument. `elements` may be a Component, Polygon, or list of Components.

    Args:
        elements: Component(/Reference), list of Component(/Reference), or Polygon
            Polygons to outline or Component containing polygons to outline.
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
    layer = gf.get_layer(layer)
    gds_layer, gds_datatype = layer

    D = Component()
    if not isinstance(elements, list):
        elements = [elements]
    port_list = []
    for e in elements:
        if isinstance(e, Component):
            D.add_ref(e)
            port_list += list(e.ports.values())
        else:
            D.add(e)

    D_bloated = offset(
        D,
        distance=distance,
        join_first=join_first,
        num_divisions=num_divisions,
        precision=precision,
        max_points=max_points,
        join=join,
        tolerance=tolerance,
        layer=layer,
    )

    Trim = Component()
    if open_ports is not False:
        trim_width = 0 if open_ports is True else open_ports * 2
        for port in port_list:
            trim = compass(size=(distance + 6 * precision, port.width + trim_width))
            trim_ref = Trim << trim
            trim_ref.connect("E", port, overlap=2 * precision)

    Outline = boolean(
        A=D_bloated,
        B=[D, Trim],
        operation="A-B" if distance > 0 else "B-A",
        num_divisions=num_divisions,
        max_points=max_points,
        precision=precision,
        layer=layer,
    )
    if open_ports is not False and len(elements) == 1:
        for port in port_list:
            Outline.add_port(port=port)
    return Outline


def test_outline() -> None:
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = outline([e1, e2])
    assert int(c.area()) == 52


if __name__ == "__main__":
    e1 = gf.components.ellipse(radii=(6, 6))
    e2 = gf.components.ellipse(radii=(10, 4))
    c = outline([e1, e2])
    c.show(show_ports=True)
