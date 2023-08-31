import numpy as np

import gdsfactory as gf
from gdsfactory.components.cutback_component import cutback_component
from gdsfactory.components.spiral_inner_io import spiral_inner_io
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


def cutback_loss(
    cutback_function: ComponentFactory = cutback_component,
    loss: tuple[float, ...] = (1 + 1 * i for i in range(3)),
    loss_dB: float = 10e-3,
    cols: int | None = 4,
    rows: int | None = None,
    **kwargs,
) -> list[gf.Component]:
    """Returns a list of component cutbacks.

    Args:
        cutback_function: cutback function.
        loss: list of target loss in dB.
        loss_dB: loss per component.
        cols: number of columns.
        rows: number of rows.
        kwargs: additional cutback arguments.
    """
    loss = np.array(list(loss))

    if rows and cols:
        raise ValueError("Specify either cols or rows")
    elif rows is None:
        rows_list = loss / loss_dB / cols
        return [
            cutback_function(rows=int(rows), cols=cols, **kwargs) for rows in rows_list
        ]
    elif cols is None:
        cols_list = loss / loss_dB / rows
        return [
            cutback_function(rows=rows, cols=int(cols), **kwargs) for cols in cols_list
        ]
    else:
        raise ValueError("Specify either cols or rows")


def cutback_loss_spirals(
    spiral: ComponentFactory = spiral_inner_io,
    loss: tuple[float, ...] = (4 + 3 * i for i in range(3)),
    cross_section: CrossSectionSpec = "strip",
    loss_dB_per_m: float = 300,
    **kwargs,
) -> list[gf.Component]:
    """Returns a list of spirals.

    Args:
        spiral: spiral factory.
        loss: list of target loss in dB.
        cross_section: strip or rib.
        loss_dB_per_m: loss per meter.
        **kwargs: additional spiral arguments.
    """
    lengths = [loss_dB / loss_dB_per_m * 1e6 for loss_dB in loss]
    return [
        spiral(length=length, cross_section=cross_section, **kwargs)
        for length in lengths
    ]


if __name__ == "__main__":
    # c = gf.pack(
    #     cutback_loss_spirals(
    #         decorator=gf.c.add_grating_couplers_with_loopback_fiber_array
    #     )
    # )[0]
    c = gf.pack(cutback_loss(decorator=gf.routing.add_fiber_array))[0]
    c.show()
