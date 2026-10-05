from __future__ import annotations

__all__ = ["gear"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def gear(
    n_teeth: int = 20,
    module_size: float = 2.0,
    pressure_angle: float = 20.0,
    hub_radius: float | None = None,
    hub_hole_radius: float = 0.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a gear with simplified trapezoidal teeth.

    Args:
        n_teeth: number of teeth.
        module_size: gear module (pitch diameter / number of teeth).
        pressure_angle: pressure angle in degrees.
        hub_radius: radius of the central hub disc. Defaults to root_radius * 0.6.
        hub_hole_radius: radius of a center hole (0 to disable).
        layer: layer spec.
    """
    return cf.gear(
        n_teeth=n_teeth,
        module_size=module_size,
        pressure_angle=pressure_angle,
        hub_radius=hub_radius,
        hub_hole_radius=hub_hole_radius,
        layer=layer,
    )


if __name__ == "__main__":
    c = gear()
    c.show()
