from __future__ import annotations

__all__ = ["litho_calipers"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size


@gf.cell_with_module_name(tags=["pcms"])
def litho_calipers(
    notch_size: Size = (2.0, 5.0),
    notch_spacing: float = 2.0,
    num_notches: int = 11,
    offset_per_notch: float = 0.1,
    row_spacing: float = 0.0,
    layer1: LayerSpec = "WG",
    layer2: LayerSpec = "SLAB150",
) -> Component:
    """Vernier caliper structure to test lithography alignment.

    Only the middle finger is aligned and the rest are offset.
    adapted from phidl

    Args:
        notch_size: [xwidth, yheight].
        notch_spacing: in um.
        num_notches: number of notches.
        offset_per_notch: in um.
        row_spacing: 0
        layer1: layer.
        layer2: layer.
    """
    return cf.litho_calipers(
        notch_size=notch_size,
        notch_spacing=notch_spacing,
        num_notches=num_notches,
        offset_per_notch=offset_per_notch,
        row_spacing=row_spacing,
        layer1=layer1,
        layer2=layer2,
    )


if __name__ == "__main__":
    c = litho_calipers()
    c.show()
