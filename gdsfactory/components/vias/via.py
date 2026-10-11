from __future__ import annotations

__all__ = ["via", "via1", "via2", "via_circular", "viac"]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import LayerSpec, Size


@gf.cell_with_module_name(tags=["vias"])
def via(
    size: Size = (0.7, 0.7),
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    bbox_layers: Sequence[LayerSpec] | None = None,
    bbox_offset: float = 0,
    bbox_offsets: Sequence[float] | None = None,
    pitch: float = 2,
    column_pitch: float | None = None,
    row_pitch: float | None = None,
) -> Component:
    """Rectangular via.

    Args:
        size: in x and y direction.
        enclosure: inclusion of via.
        layer: via layer.
        bbox_layers: layers for the bounding box.
        bbox_offset: in um.
        bbox_offsets: List of offsets for each bbox_layer.
        pitch: pitch between vias.
        column_pitch: Optional pitch between columns of vias. Default is pitch.
        row_pitch: Optional pitch between rows of vias. Default is pitch.

    ```text
        enclosure
        _________________________________________
        |<--->                                  |
        |             gap[0]    size[0]         |
        |             <------> <----->          |
        |      ______          ______           |
        |     |      |        |      |          |
        |     |      |        |      |  size[1] |
        |     |______|        |______|          |
        |      <------------->                  |
        |           pitch                       |
        |_______________________________________|
    ```
    """
    return cf.via(
        size=size,
        enclosure=enclosure,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offset=bbox_offset,
        bbox_offsets=bbox_offsets,
        pitch=pitch,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )


@gf.cell_with_module_name(tags=["vias"])
def via_circular(
    radius: float = 0.35,
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    pitch: float | None = 2,
    column_pitch: float | None = None,
    row_pitch: float | None = None,
    angle_resolution: float = 2.5,
) -> Component:
    """Circular via.

    Args:
        radius: in um.
        enclosure: inclusion of via in um for the layer above.
        layer: via layer.
        pitch: pitch between vias.
        column_pitch: Optional pitch between columns of vias. Default is pitch.
        row_pitch: Optional pitch between rows of vias. Default is pitch.
        angle_resolution: number of degrees per point.
    """
    return cf.via_circular(
        radius=radius,
        enclosure=enclosure,
        layer=layer,
        pitch=pitch,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
        angle_resolution=angle_resolution,
    )


viac = CellAlias(via, layer="VIAC")
via1 = CellAlias(via, layer="VIA1", enclosure=1)
via2 = CellAlias(via, layer="VIA2")
