from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec

_cols_200mm_wafer = (2, 6, 6, 8, 8, 6, 6, 2)


@gf.cell
def wafer(
    reticle: ComponentSpec = "die",
    cols: tuple[int, ...] = _cols_200mm_wafer,
    xspacing: float | None = None,
    yspacing: float | None = None,
    die_name_col_row: bool = False,
) -> Component:
    """Returns complete wafer. Useful for mask aligner steps.

    Args:
        reticle: spec for each wafer reticle.
        cols: how many columns per row.
        xspacing: optional spacing, defaults to reticle.dxsize.
        yspacing: optional spacing, defaults to reticle.dysize.
        die_name_col_row: if True, die name is row_col, otherwise is a number
    """
    c = gf.Component()
    die = gf.get_component(reticle)
    xspacing = xspacing or die.dxsize
    yspacing = yspacing or die.dysize

    i = 1
    for col in range(len(cols)):
        for row in range(cols[col]):
            die_name = f"{col+1}_{row+1}" if die_name_col_row else str(i)
            die = gf.get_component(reticle, die_name=die_name)
            ref = c.add_ref(die)
            ref.dmovex((row - cols[col] / 2) * xspacing)
            ref.dmovey(col * yspacing)
            i += 1

    return c


if __name__ == "__main__":
    c = wafer(die_name_col_row=True)
    c.show()
