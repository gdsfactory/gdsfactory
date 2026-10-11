from __future__ import annotations

__all__ = ["spiral_fermat"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_fermat(
    width: float = 1.0,
    n_turns: int = 5,
    a: float = 5.0,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a Fermat spiral: r = a * sqrt(theta).

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        a: scaling factor.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    return cf.spiral_fermat(
        width=width,
        n_turns=n_turns,
        a=a,
        angle_resolution=angle_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = spiral_fermat()
    c.show()
