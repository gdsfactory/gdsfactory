from __future__ import annotations

__all__ = [
    "via_stack_with_offset",
    "via_stack_with_offset_m1_m3",
    "via_stack_with_offset_ppp_m1",
]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name(tags=["vias"])
def via_stack_with_offset(
    layers: LayerSpecs = ("PPP", "M1"),
    size: Size | None = (10, 10),
    sizes: Sequence[Size] | None = None,
    layer_offsets: Sequence[float] | None = None,
    vias: Sequence[ComponentSpec | None] = (None, "viac"),
    offsets: Sequence[float] | None = None,
    layer_to_port_orientations: dict[LayerSpec, list[int]] | None = None,
) -> Component:
    """Rectangular layer transition with offset between layers.

    Args:
        layers: layer specs between vias.
        size: for all vias array.
        sizes: Optional size for each via array. Overrides size.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: via spec for previous layer. None for no via.
        offsets: optional offset for each layer relatively to the previous one.
            By default it only offsets by size[1] if there is a via.
        layer_to_port_orientations: Optional dictionary with layer to port orientations.

        side view

    ```text
         __________________________
        |                          |
        |                          | layers[2]
        |__________________________|           vias[2] = None
        |                          |
        | layer_offsets[1]+size    | layers[1]
        |__________________________|
            |     |
            vias[1]
         ___|_____|__
        |            |
        |  sizes[0]  |  layers[0]
        |____________|
    ```

            vias[0] = None

    """
    return cf.via_stack_with_offset(
        layers=layers,
        size=size,
        sizes=sizes,
        layer_offsets=layer_offsets,
        vias=vias,
        offsets=offsets,
        layer_to_port_orientations=layer_to_port_orientations,
    )


via_stack_with_offset_ppp_m1 = CellAlias(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, "viac"),
)

via_stack_with_offset_ppp_m1 = CellAlias(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, "viac"),
)

via_stack_with_offset_m1_m3 = CellAlias(
    via_stack_with_offset,
    layers=("M1", "M2", "MTOP"),
    vias=(None, "via1", "via2"),
)
