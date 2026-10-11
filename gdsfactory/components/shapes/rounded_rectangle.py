from __future__ import annotations

__all__ = ["rounded_rectangle"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def rounded_rectangle(
    width: float = 20.0,
    height: float = 10.0,
    corner_radius_x: float = 3.0,
    corner_radius_y: float | None = None,
    n_corner_points: int = 20,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a rectangle with rounded corners, centered at origin.

    Args:
        width: total width of the rectangle.
        height: total height of the rectangle.
        corner_radius_x: x-radius of the corner arcs.
        corner_radius_y: y-radius of the corner arcs. Defaults to corner_radius_x.
        n_corner_points: number of points per corner arc.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.rounded_rectangle(
        width=width,
        height=height,
        corner_radius_x=corner_radius_x,
        corner_radius_y=corner_radius_y,
        n_corner_points=n_corner_points,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = rounded_rectangle()
    c.show()
