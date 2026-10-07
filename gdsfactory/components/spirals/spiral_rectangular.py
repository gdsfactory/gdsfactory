from __future__ import annotations

__all__ = ["spiral_rectangular"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_rectangular(
    n_turns: int = 4,
    width: float = 1.0,
    start_length: float = 10.0,
    pitch: float = 3.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a rectangular/Manhattan spiral as a polygon.

    Each segment grows by pitch per half-turn, building an expanding
    rectangular spiral. The outline is created with the given width offset.

    Args:
        n_turns: number of full turns.
        width: width of the spiral trace.
        start_length: initial segment length.
        pitch: spacing between adjacent turns (center-to-center).
        layer: layer spec.
    """
    return cf.spiral_rectangular(
        n_turns=n_turns,
        width=width,
        start_length=start_length,
        pitch=pitch,
        layer=layer,
    )


if __name__ == "__main__":
    c = spiral_rectangular()
    c.show()
