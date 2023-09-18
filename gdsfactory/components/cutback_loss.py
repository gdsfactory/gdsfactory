from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.components.cutback_component import cutback_component
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.spiral_inner_io import spiral_inner_io
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


def cutback_loss(
    component: ComponentFactory = mmi1x2,
    cutback: ComponentFactory = cutback_component,
    loss: tuple[float, ...] = tuple(1 + 1 * i for i in range(3)),
    loss_dB: float = 10e-3,
    cols: int | None = 4,
    rows: int | None = None,
    **kwargs,
) -> list[gf.Component]:
    """Returns a list of component cutbacks.

    Args:
        component: component factory.
        cutback: cutback function.
        loss: list of target loss in dB.
        loss_dB: loss per component.
        cols: number of columns.
        rows: number of rows.

    Keyword Args:
        port1: name of first optical port.
        port2: name of second optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        mirror1: mirrors first component.
        mirror2: mirrors second component.
        straight_length: length of the straight section between cutbacks.
        straight_length_pair: length of the straight section between each component pair.
        cross_section: specification (CrossSection, string or dict).
        kwargs: component settings.

    """
    loss = np.array(list(loss))

    if rows and cols:
        raise ValueError("Specify either cols or rows")
    elif rows is None:
        rows_list = loss / loss_dB / cols
        rows_list = rows_list // 2 * 2 + 1
        return [
            cutback(component=component, rows=int(rows), cols=cols, **kwargs)
            for rows in rows_list
        ]
    elif cols is None:
        cols_list = loss / loss_dB / rows
        return [
            cutback(component=component, rows=rows, cols=int(cols), **kwargs)
            for cols in cols_list
        ]
    else:
        raise ValueError("Specify either cols or rows")


def cutback_loss_spirals(
    spiral: ComponentFactory = spiral_inner_io,
    loss: tuple[float, ...] = tuple(4 + 3 * i for i in range(3)),
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
        kwargs: additional spiral arguments.
    """
    lengths = [loss_dB / loss_dB_per_m * 1e6 for loss_dB in loss]
    return [
        spiral(length=length, cross_section=cross_section, **kwargs)
        for length in lengths
    ]


cutback_loss_mmi1x2 = partial(cutback_loss, component=mmi1x2, port2="o3", mirror2=True)


if __name__ == "__main__":
    # c = gf.pack(
    #     cutback_loss_spirals(
    #         decorator=gf.c.add_grating_couplers_with_loopback_fiber_array
    #     )
    # )[0]
    # components = cutback_loss(
    #     component=gf.c.mmi2x2, decorator=gf.routing.add_fiber_array
    # )
    components = cutback_loss_mmi1x2(decorator=gf.routing.add_fiber_array)
    c = gf.pack(components)[0]
    c.show()
