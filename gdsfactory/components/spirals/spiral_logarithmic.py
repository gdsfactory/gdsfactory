from __future__ import annotations

__all__ = ["spiral_logarithmic"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_logarithmic(
    width: float = 0.5,
    n_turns: int = 4,
    a: float = 1.0,
    b: float = 0.1,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a logarithmic spiral: r = a * exp(b * theta).

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        a: initial radius scaling factor.
        b: growth rate.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    return cf.spiral_logarithmic(
        width=width,
        n_turns=n_turns,
        a=a,
        b=b,
        angle_resolution=angle_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = spiral_logarithmic()
    c.show()
