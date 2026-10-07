from __future__ import annotations

__all__ = ["folded_spring"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def folded_spring(
    beam_width: float = 0.5,
    beam_length: float = 20.0,
    n_folds: int = 4,
    fold_gap: float = 1.0,
    anchor_width: float = 5.0,
    anchor_length: float = 3.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a folded flexure spring (serpentine meander).

    Alternating horizontal beams connected at their ends forming a
    serpentine pattern. Starts at a bottom anchor and ends at a top anchor.

    Args:
        beam_width: width of each beam segment.
        beam_length: length of each horizontal beam segment.
        n_folds: number of horizontal beam segments.
        fold_gap: vertical gap between adjacent beams.
        anchor_width: width (horizontal) of the anchor pads.
        anchor_length: length (vertical) of the anchor pads.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.folded_spring(
        beam_width=beam_width,
        beam_length=beam_length,
        n_folds=n_folds,
        fold_gap=fold_gap,
        anchor_width=anchor_width,
        anchor_length=anchor_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = folded_spring()
    c.show()
