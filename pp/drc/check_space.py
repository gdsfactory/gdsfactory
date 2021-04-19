from typing import Tuple

import klayout.db as pya

from pp.component import Component
from pp.types import ComponentOrPath


def check_space(
    gdspath: ComponentOrPath,
    layer: Tuple[int, int] = (1, 0),
    min_space: float = 0.150,
    dbu: float = 1e3,
    ignore_angle_deg: int = 80,
    whole_edges: bool = False,
    metrics: None = None,
    min_projection: None = None,
    max_projection: None = None,
) -> int:
    """Reads layer from top cell and returns a the area that violates min space.

    If "whole_edges" is true, the resulting EdgePairs collection will receive the whole edges which contribute in the space check.

    "metrics" can be one of the constants Euclidian, Square or Projection. See there for a description of these constants.
    Use nil for this value to select the default (Euclidian metrics).

    "ignore_angle" specifies the angle limit of two edges. If two edges form an angle equal or above the given value, they will not contribute in
    the check. Setting this value to 90 (the default) will exclude edges with an angle of 90 degree or more from the check.
    Use nil for this value to select the default.

    "min_projection" and "max_projection" allow selecting edges by their projected value upon each other. It is sufficient if the projection of on
    e edge on the other matches the specified condition. The projected length must be larger or equal to "min_projection" and less than "max_proje
    ction". If you don't want to specify one limit, pass nil to the respective value.

    Args:
        gdspath: path to GDS or Component
        layer: tuple
        min_space: in um
        dbu: database units (1000 um/nm)
        ignore_angle_deg: The angle above which no check is performed
        other: The other region against which to check
        whole_edges: If true, deliver the whole edges
        metrics: Specify the metrics type
        min_projection The lower threshold of the projected length of one edge onto another
        max_projection The upper limit of the projected length of one edge onto another

    """

    if isinstance(gdspath, Component):
        gdspath.flatten()
        gdspath = gdspath.write_gds()
    layout = pya.Layout()
    layout.read(str(gdspath))
    cell = layout.top_cell()
    region = pya.Region(cell.begin_shapes_rec(layout.layer(layer[0], layer[1])))

    d = region.space_check(
        min_space * dbu,
        whole_edges,
        metrics,
        ignore_angle_deg,
        min_projection,
        max_projection,
    )
    # print(d.polygons().area())
    return d.polygons().area()


if __name__ == "__main__":
    import pp

    space = 0.12
    min_space = 0.1
    dbu = 1000
    layer = pp.LAYER.WG
    gdspath = pp.components.straight_array(spacing=space)
    pp.show(gdspath)

    if isinstance(gdspath, Component):
        gdspath.flatten()
        gdspath = gdspath.write_gds()
    layout = pya.Layout()
    layout.read(str(gdspath))
    cell = layout.top_cell()
    region = pya.Region(cell.begin_shapes_rec(layout.layer(layer[0], layer[1])))
    # print(region.polygons().area())
    # d = region.space_check(min_space * dbu, False, "Square", 80, None, None)
