from __future__ import annotations

__all__ = ["circle_wave"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def circle_wave(
    radius: float = 10.0,
    amplitude: float = 1.0,
    n_oscillations: int = 8,
    angle_resolution: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a circle with a sinusoidal boundary variation.

    The boundary radius varies as r(theta) = radius + amplitude * sin(n * theta).

    Args:
        radius: mean radius.
        amplitude: amplitude of sinusoidal modulation.
        n_oscillations: number of oscillations around the boundary.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    return cf.circle_wave(
        radius=radius,
        amplitude=amplitude,
        n_oscillations=n_oscillations,
        angle_resolution=angle_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = circle_wave()
    c.show()
