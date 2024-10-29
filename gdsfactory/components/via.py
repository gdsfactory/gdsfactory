from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def via(
    size: tuple[float, float] = (0.7, 0.7),
    spacing: tuple[float, float] | None = (2.0, 2.0),
    gap: tuple[float, float] | None = None,
    enclosure: float = 1.0,
    layer: LayerSpec = "VIAC",
    bbox_layers: tuple[LayerSpec, ...] | None = None,
    bbox_offset: float = 0,
    bbox_offsets: tuple[float, ...] | None = None,
) -> Component:
    """Rectangular via.

    Defaults to a square via.

    Args:
        size: in x, y direction.
        spacing: pitch_x, pitch_y.
        gap: edge to edge via gap in x, y.
        enclosure: inclusion of via.
        layer: via layer.
        bbox_layers: layers for the bounding box.
        bbox_offset: in um.
        bbox_offsets: List of offsets for each bbox_layer.

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
        |         spacing[0]                    |
        |_______________________________________|
    """
    if spacing is None and gap is None:
        raise ValueError("either spacing or gap should be defined")
    elif spacing is not None and gap is not None:
        raise ValueError("You can't define spacing and gap at the same time")
    if spacing is None:
        spacing = (size[0] + gap[0], size[1] + gap[1])

    c = Component()
    c.info["xspacing"] = spacing[0]
    c.info["yspacing"] = spacing[1]
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


viac = partial(via, layer="VIAC")
via1 = partial(via, layer="VIA1", enclosure=2)
via2 = partial(via, layer="VIA2")


if __name__ == "__main__":
    c = via()
    # c.pprint()
    print(c)
    c.show()
