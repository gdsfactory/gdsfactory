from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import Component, ComponentSpec

_cols_200mm_wafer = (2, 6, 6, 8, 8, 6, 6, 2)


@gf.cell
def wafer(
    reticle: ComponentSpec = "die",
    cols: tuple[int, ...] = _cols_200mm_wafer,
    xspacing: float | None = None,
    yspacing: float | None = None,
) -> Component:
    """Returns complete wafer. Useful for mask aligner steps.

    Args:
        reticle: spec for each wafer reticle.
        cols: how many columns per row.
        xspacing: optional spacing, defaults to reticle.xsize.
        yspacing: optional spacing, defaults to reticle.ysize.
    """
    c = gf.Component()
    reticle = gf.get_component(reticle)
    xspacing = xspacing or reticle.xsize
    yspacing = yspacing or reticle.ysize

    for i, columns in enumerate(cols):
        ref = c.add_array(
            reticle, rows=1, columns=columns, spacing=(xspacing, yspacing)
        )
        ref.x = 0
        ref.movey(i * yspacing)

    return c


if __name__ == "__main__":
    c = wafer()
    c.show()
