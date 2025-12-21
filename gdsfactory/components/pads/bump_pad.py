from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import (
    LayerSpec,
)

from ..shapes import octagon
from ..vias import via_stack

__all__ = ["bump_pad", "bump_pad_grid"]


@gf.cell_with_module_name
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
    c = Component()
    layer = gf.get_layer(layer)
    size_ = gf.get_constant(size)
    rect = octagon(
        side_length=size, layer=layer, port_width=port_width, port_type=port_type
    )
    c_ref = c.add_ref(rect)
    if add_via:
        via_n = c << via_stack()
        via_n.y = c_ref.ports["o5"].y - via_n.ports["e1"].width / 2
        p = via_n.ports["e2"]
        c.add_port(
            name="e2",
            center=p.center,
            width=p.width,
            orientation=p.orientation,
            port_type=p.port_type,
            layer=port_layer,
        )

        via_e = c << via_stack()
        via_e.x = c_ref.ports["o3"].x - via_e.ports["e1"].width / 2
        p = via_e.ports["e3"]
        c.add_port(
            name="e3",
            center=p.center,
            width=p.width,
            orientation=p.orientation,
            port_type=p.port_type,
            layer=port_layer,
        )

        via_s = c << via_stack()
        via_s.y = c_ref.ports["o1"].y + via_s.ports["e1"].width / 2
        p = via_s.ports["e4"]
        c.add_port(
            name="e4",
            center=p.center,
            width=p.width,
            orientation=p.orientation,
            port_type=p.port_type,
            layer=port_layer,
        )

        via_w = c << via_stack()
        via_w.x = c_ref.ports["o7"].x + via_w.ports["e1"].width / 2
        p = via_w.ports["e1"]
        c.add_port(
            name="e1",
            center=p.center,
            width=p.width,
            orientation=p.orientation,
            port_type=p.port_type,
            layer=port_layer,
        )
    else:
        for i, j in enumerate([1, 3, 5, 7]):
            p = c_ref.ports[f"o{j}"]
            c.add_port(
                name=f"e{i + 1}",
                center=p.center,
                width=port_width,
                orientation=p.orientation,
                port_type=port_type,
                layer=layer,
            )
    c.info["size"] = size_

    return c


@gf.cell_with_module_name
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
    c = Component()

    pad_kwargs: dict[str, Any] = {}
    if layer is not None:
        pad_kwargs["layer"] = layer
    if size is not None:
        pad_kwargs["size"] = size

    pad_component = bump_pad(
        size=size,
        layer=layer,
        port_width=port_width,
        port_layer=port_layer,
        add_via=add_via,
    )

    for col in range(columns):
        for row in range(rows):
            if skip_pads is not None and (col, row) in skip_pads:
                continue
            pad = c << pad_component
            center = (col * column_pitch, row * row_pitch + (col % 2) * offset)
            pad.center = center
            c.add_ports(pad.ports, prefix=f"e{row + 1}_{col + 1}_")

    if auto_rename_ports:
        c.auto_rename_ports()
    return c
