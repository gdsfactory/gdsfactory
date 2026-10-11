from __future__ import annotations

__all__ = ["rect_taper"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def rect_taper(
    rect_width: float = 1.0,
    rect_length: float = 10.0,
    taper_length: float = 5.0,
    taper_width: float = 4.0,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a rectangle connected to a linear taper.

    The rectangle of width rect_width and length rect_length is connected
    on the right to a taper that linearly expands from rect_width to
    taper_width over taper_length.

    Args:
        rect_width: width of the rectangular section.
        rect_length: length of the rectangular section.
        taper_length: length of the taper section.
        taper_width: end width of the taper.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.rect_taper(
        rect_width=rect_width,
        rect_length=rect_length,
        taper_length=taper_length,
        taper_width=taper_width,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = rect_taper()
    c.show()
