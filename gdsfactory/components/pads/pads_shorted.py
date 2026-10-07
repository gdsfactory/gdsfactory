from __future__ import annotations

__all__ = ["pads_shorted"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell_with_module_name(tags=["pads"])
def pads_shorted(
    pad: ComponentSpec = "pad",
    columns: int = 8,
    pad_pitch: float = 150.0,
    layer_metal: LayerSpec = "MTOP",
    metal_width: float = 10,
) -> Component:
    """Returns a 1D array of shorted_pads.

    Args:
        pad: pad spec.
        columns: number of columns.
        pad_pitch: in um
        layer_metal: for the short.
        metal_width: for the short.
    """
    return cf.pads_shorted(
        pad=pad,
        columns=columns,
        pad_pitch=pad_pitch,
        layer_metal=layer_metal,
        metal_width=metal_width,
    )
