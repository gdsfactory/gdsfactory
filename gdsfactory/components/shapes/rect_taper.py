from __future__ import annotations

__all__ = ["rect_taper"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    c = Component()
    hw = rect_width / 2
    tw = taper_width / 2
    total_length = rect_length + taper_length

    points = [
        (0, -hw),
        (rect_length, -hw),
        (total_length, -tw),
        (total_length, tw),
        (rect_length, hw),
        (0, hw),
    ]
    c.add_polygon(points, layer=layer)

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            center=(0, 0),
            width=rect_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(total_length, 0),
            width=taper_width,
            orientation=0,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = rect_taper()
    c.show()
