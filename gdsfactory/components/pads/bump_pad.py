from __future__ import annotations

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import (
    LayerSpec,
)

__all__ = ["bump_pad", "bump_pad_grid"]


@gf.cell_with_module_name(tags=["pads"])
def bump_pad(
    size: float = 36.244,
    layer: LayerSpec = "MTOP",
    port_width: float = 10.0,
    port_layer: LayerSpec = "M2",
    port_type: str = "pad",
    add_via: bool = True,
) -> Component:
    """Returns rectangular pad with ports.

    Args:
        size: octagon edge of octagon.
        layer: bump pad layer.
        port_width: width of the port for electrical routing.
        port_layer: layer of the port for electrical routing.
        port_type: port type for pad port.
        add_via: whether to add a via stack.
    """
    return cf.bump_pad(
        size=size,
        layer=layer,
        port_width=port_width,
        port_layer=port_layer,
        port_type=port_type,
        add_via=add_via,
    )


@gf.cell_with_module_name(tags=["pads"])
def bump_pad_grid(
    columns: int = 6,
    rows: int = 6,
    column_pitch: float = 121.89,
    row_pitch: float = 132.66,
    offset: float = 66.33,
    port_width: float = 10,
    port_layer: LayerSpec = "M2",
    size: float = 36.244,
    layer: LayerSpec = "MTOP",
    auto_rename_ports: bool = False,
    skip_pads: list[tuple[int, int]] | None = None,
    add_via: bool = True,
) -> Component:
    """Returns 2D array of bump pads.

    Args:
        columns: number of columns.
        rows: number of rows.
        column_pitch: x pitch.
        row_pitch: y pitch.
        offset: offset for alternating columns.
        port_width: width of the port for electrical routing.
        port_layer: layer of the port for electrical routing.
        size: pad size.
        layer: bump pad layer.
        auto_rename_ports: True to auto rename ports.
        skip_pads: list of (col, row) tuples to skip.
        add_via: whether to add a via stack.
    """
    return cf.bump_pad_grid(
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
        offset=offset,
        port_width=port_width,
        port_layer=port_layer,
        size=size,
        layer=layer,
        auto_rename_ports=auto_rename_ports,
        skip_pads=skip_pads,
        add_via=add_via,
    )
