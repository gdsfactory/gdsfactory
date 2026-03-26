from __future__ import annotations

__all__ = ["vernier_scale"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def vernier_scale(
    n_divisions: int = 10,
    pitch_main: float = 10.0,
    pitch_vernier: float = 9.8,
    mark_width: float = 1.0,
    mark_height_main: float = 20.0,
    mark_height_vernier: float = 15.0,
    layer_main: LayerSpec = "WG",
    layer_vernier: LayerSpec = (2, 0),
) -> Component:
    """Vernier scale for overlay measurement.

    Creates two rows of rectangular marks: a main scale and a vernier scale.
    The slight pitch difference between the two scales allows sub-pitch
    overlay measurement. The zero marks of both scales are centered at the
    origin.

    Args:
        n_divisions: Number of marks on each side of the center mark
            (total marks per scale = 2 * n_divisions + 1).
        pitch_main: Center-to-center spacing of main scale marks in um.
        pitch_vernier: Center-to-center spacing of vernier scale marks in um.
        mark_width: Width of each rectangular mark in um.
        mark_height_main: Height of each main scale mark in um.
        mark_height_vernier: Height of each vernier scale mark in um.
        layer_main: Layer specification for the main scale marks.
        layer_vernier: Layer specification for the vernier scale marks.
    """
    c = Component()

    hw = mark_width / 2

    # Main scale marks (below y=0)
    for i in range(-n_divisions, n_divisions + 1):
        x = i * pitch_main
        c.add_polygon(
            [
                (x - hw, 0),
                (x + hw, 0),
                (x + hw, -mark_height_main),
                (x - hw, -mark_height_main),
            ],
            layer=layer_main,
        )

    # Vernier scale marks (above y=0)
    for i in range(-n_divisions, n_divisions + 1):
        x = i * pitch_vernier
        c.add_polygon(
            [
                (x - hw, 0),
                (x + hw, 0),
                (x + hw, mark_height_vernier),
                (x - hw, mark_height_vernier),
            ],
            layer=layer_vernier,
        )

    return c


if __name__ == "__main__":
    c = vernier_scale()
    c.show()
