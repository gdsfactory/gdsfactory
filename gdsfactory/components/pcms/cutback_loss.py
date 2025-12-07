__all__ = [
    "cutback_loss",
    "cutback_loss_bend90",
    "cutback_loss_bend180",
    "cutback_loss_mmi1x2",
    "cutback_loss_spirals",
]

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


def cutback_loss(
    component: ComponentSpec = "mmi1x2",
    cutback: ComponentSpec = "cutback_component",
    loss: Sequence[float] = (1.0, 2.0, 3.0),
    loss_dB: float = 10e-3,
    cols: int | None = 4,
    rows: int | None = None,
    **kwargs: Any,
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
    loss_array = np.array(loss)

    if rows is not None and cols is not None:
        raise ValueError("Specify either 'cols' or 'rows', but not both.")

    if cols is not None:
        # Calculate rows for each target loss
        rows_array = (loss_array / loss_dB) / cols
        rows_list = [int(np.ceil(rows) // 2 * 2 + 1) for rows in rows_array]
        settings = dict(
            component=component,
            cutback=cutback,
            cols=cols,
            **kwargs,
        )
        return [
            gf.get_component(cutback, settings=settings, rows=rows)
            for rows in rows_list
        ]
    if rows is not None:
        # Calculate cols for each target loss
        cols_array = (loss_array / loss_dB) / rows
        cols_list = [int(np.ceil(cols)) for cols in cols_array]
        settings = dict(
            component=component,
            cutback=cutback,
            rows=rows,
            **kwargs,
        )
        return [
            gf.get_component(cutback, settings=settings, cols=cols)
            for cols in cols_list
        ]
    raise ValueError("You must specify either 'cols' or 'rows'.")


_loss_default = tuple(4 + 3 * i for i in range(3))


def cutback_loss_spirals(
    spiral: ComponentSpec = "spiral",
    loss: Sequence[float] = _loss_default,
    cross_section: CrossSectionSpec = "strip",
    loss_dB_per_m: float = 300,
    **kwargs: Any,
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
        gf.get_component(spiral, length=length, cross_section=cross_section, **kwargs)
        for length in lengths
    ]


cutback_loss_mmi1x2 = partial(
    cutback_loss, component="mmi1x2", port2="o3", mirror2=True
)
cutback_loss_bend90 = partial(
    cutback_loss, component="bend_euler", cutback="cutback_bend90", cols=12
)
cutback_loss_bend180 = partial(
    cutback_loss, component="bend_euler180", cutback="cutback_bend180", cols=12
)


def cutback_loss_mmi2x2(
    component: gf.typings.ComponentSpec,
    loss: tuple[float, ...] = (1.0, 2.0, 3.0),
    loss_dB: float = 10e-3,
    cols: int | None = 4,
    rows: int | None = None,
    **kwargs: Any,
) -> list[gf.Component]:
    """Returns a list of component cutbacks for 2x2 MMI.

    Args:
        component: component to test.
        loss: list of target loss in dB.
        loss_dB: loss per component.
        cols: number of columns.
        rows: number of rows.
        kwargs: additional kwargs for cutback_2x2.
    """
    loss_array = np.array(loss)

    if rows is not None and cols is not None:
        raise ValueError("Specify either 'cols' or 'rows', but not both.")

    if cols is not None:
        # Calculate rows for each target loss
        rows_array = (loss_array / loss_dB) / cols
        rows_list = [int(np.ceil(rows) // 2 * 2 + 1) for rows in rows_array]
        return [
            gf.components.cutback_2x2(
                component=component,
                cols=cols,
                rows=rows,
                **kwargs,
            )
            for rows in rows_list
        ]
    if rows is not None:
        # Calculate cols for each target loss
        cols_array = (loss_array / loss_dB) / rows
        cols_list = [int(np.ceil(cols)) for cols in cols_array]
        return [
            gf.components.cutback_2x2(
                component=component,
                rows=rows,
                cols=cols,
                **kwargs,
            )
            for cols in cols_list
        ]
    raise ValueError("You must specify either 'cols' or 'rows'.")
