from __future__ import annotations

__all__ = [
    "pad",
    "pad_array",
    "pad_array0",
    "pad_array90",
    "pad_array180",
    "pad_array270",
    "pad_rectangular",
    "pad_small",
]

from functools import partial

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import (
    AngleInDegrees,
    ComponentSpec,
    Float2,
    Ints,
    LayerSpec,
    Size,
)

from .._schematic import pad_schematic


@gf.cell_with_module_name(schematic_function=pad_schematic, tags=["pads"])
def pad(
    size: Size | str = (100.0, 100.0),
    layer: LayerSpec = "MTOP",
    bbox_layers: tuple[LayerSpec, ...] | None = None,
    bbox_offsets: tuple[float, ...] | None = None,
    port_inclusion: float = 0,
    port_orientation: AngleInDegrees | None = 0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    port_type: str = "pad",
) -> Component:
    """Returns rectangular pad with ports.

    Args:
        size: x, y size.
        layer: pad layer.
        bbox_layers: list of layers.
        bbox_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        port_inclusion: from edge.
        port_orientation: in degrees for the center port.
        port_orientations: list of port_orientations to add. None does not add ports.
        port_type: port type for pad port.
    """
    return cf.pad(
        size=size,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
        port_orientations=port_orientations,
        port_type=port_type,
    )


pad_rectangular = partial(pad, size="pad_size")
pad_small = partial(pad, size=(80, 80))


@gf.cell_with_module_name(schematic_function=pad_schematic, tags=["pads"])
def pad_array(
    pad: ComponentSpec = "pad",
    columns: int = 6,
    rows: int = 1,
    column_pitch: float = 150.0,
    row_pitch: float = 150.0,
    port_orientation: AngleInDegrees = 0,
    size: Float2 | None = None,
    layer: LayerSpec | None = "MTOP",
    centered_ports: bool = False,
    auto_rename_ports: bool = False,
) -> Component:
    """Returns 2D array of pads.

    Args:
        pad: pad element.
        columns: number of columns.
        rows: number of rows.
        column_pitch: x pitch.
        row_pitch: y pitch.
        port_orientation: port orientation in deg. None for low speed DC ports.
        size: pad size.
        layer: pad layer.
        centered_ports: True add ports to center. False add ports to the edge.
        auto_rename_ports: True to auto rename ports.
    """
    return cf.pad_array(
        pad=pad,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
        port_orientation=port_orientation,
        size=size,
        layer=layer,
        centered_ports=centered_ports,
        auto_rename_ports=auto_rename_ports,
    )


pad_array90 = partial(pad_array, port_orientation=90)
pad_array270 = partial(pad_array, port_orientation=270)

pad_array0 = partial(pad_array, port_orientation=0, columns=1, rows=3)
pad_array180 = partial(pad_array, port_orientation=180, columns=1, rows=3)
