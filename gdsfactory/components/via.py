from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, Size, Spacing


@gf.cell
def via(
    size: Size = (0.7, 0.7),
    spacing: Spacing | None = None,
    gap: float | None = None,
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    bbox_layers: Sequence[LayerSpec] | None = None,
    bbox_offset: float = 0,
    bbox_offsets: Sequence[float] | None = None,
    pitch: float = 2,
) -> Component:
    """Rectangular via.

    Args:
        size: in x and y direction.
        spacing: pitch_x, pitch_y. Deprecated, use pitch instead.
        gap: edge to edge via gap in x, y. Deprecated, use pitch instead.
        enclosure: inclusion of via.
        layer: via layer.
        bbox_layers: layers for the bounding box.
        bbox_offset: in um.
        bbox_offsets: List of offsets for each bbox_layer.
        pitch: pitch between vias.

    .. code::

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
    """
    if spacing is not None:
        warnings.warn("spacing is deprecated, use pitch instead")
        pitch = spacing[0]

    if gap is not None:
        warnings.warn("gap is deprecated, use pitch instead")

    c = Component()
    c.info["pitch"] = pitch
    c.info["enclosure"] = enclosure
    c.info["xsize"] = size[0]
    c.info["ysize"] = size[1]

    width, height = size
    a = width / 2
    b = height / 2
    c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    bbox_layers = bbox_layers or []
    bbox_offsets = bbox_offsets or [bbox_offset] * len(bbox_layers)

    if len(bbox_offsets) != len(bbox_layers):
        raise ValueError(
            f"bbox_offsets {bbox_offsets=} should have the same length as bbox_layers {bbox_layers=}"
        )

    for layer, bbox_offset in zip(bbox_layers, bbox_offsets):
        a = (width + 2 * bbox_offset) / 2
        b = (height + 2 * bbox_offset) / 2
        c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)
    return c


@gf.cell
def via_circular(
    radius: float = 0.35,
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    pitch: float | None = 2,
) -> Component:
    """Circular via.

    Args:
        radius: in um.
        enclosure: inclusion of via in um for the layer above.
        layer: via layer.
        pitch: pitch between vias.
    """
    c = Component()
    _ = c << gf.c.circle(radius=radius, layer=layer)
    c.info["pitch"] = pitch
    c.info["enclosure"] = enclosure
    c.info["radius"] = radius
    c.info["xsize"] = 2 * radius
    c.info["ysize"] = 2 * radius
    return c


viac = partial(via, layer="VIAC")
via1 = partial(via, layer="VIA1", enclosure=2)
via2 = partial(via, layer="VIA2")


if __name__ == "__main__":
    c = via_circular()
    # c.pprint()
    print(c)
    c.show()
