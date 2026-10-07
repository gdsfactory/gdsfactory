from __future__ import annotations

__all__ = ["star"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def star(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    n_points: int = 5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a star shape with alternating inner and outer radii.

    Args:
        inner_radius: radius of inner vertices.
        outer_radius: radius of outer vertices.
        n_points: number of star points.
        layer: layer spec.
    """
    return cf.star(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        n_points=n_points,
        layer=layer,
    )


if __name__ == "__main__":
    c = star()
    c.show()
