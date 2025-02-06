from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell
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
    c = Component()
    pad = gf.get_component(pad)
    for i in range(columns):
        pad_ref = c.add_ref(pad)
        pad_ref.dmovex(i * pad_pitch - columns / 2 * pad_pitch + pad_pitch / 2)

    short = gf.c.rectangle(
        size=(pad_pitch * (columns - 1), metal_width),
        layer=layer_metal,
        centered=True,
    )
    c.add_ref(short)
    return c


if __name__ == "__main__":
    c = pads_shorted(metal_width=20)
    c.show()
