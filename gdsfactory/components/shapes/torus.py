from __future__ import annotations

__all__ = ["torus"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def torus(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a torus (annular sector / ring sector) centered at origin.

    Args:
        inner_radius: inner radius.
        outer_radius: outer radius.
        start_angle: start angle in degrees.
        end_angle: end angle in degrees.
        angle_resolution: degrees per arc point.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.torus(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        start_angle=start_angle,
        end_angle=end_angle,
        angle_resolution=angle_resolution,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = torus()
    c.show()
