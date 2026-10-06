from __future__ import annotations

__all__ = ["spiral_archimedes"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_archimedes(
    width: float = 1.0,
    n_turns: int = 5,
    separation: float = 2.0,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns an Archimedes spiral: r = (separation + width) / (2 * pi) * theta.

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        separation: gap between adjacent traces.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    return cf.spiral_archimedes(
        width=width,
        n_turns=n_turns,
        separation=separation,
        angle_resolution=angle_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = spiral_archimedes()
    c.show()
