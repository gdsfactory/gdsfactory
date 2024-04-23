from __future__ import annotations

import warnings
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
    bbox_layers: tuple[tuple[int, int], ...] | None = None,
    bbox_offset: float | None = None,
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
        bbox_offsets: list of offsets in um.

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
        |         xspacing                      |
        |_______________________________________|
    """
    if spacing is None and gap is None:
        raise ValueError("either spacing or gap should be defined")
    elif spacing is not None and gap is not None:
        raise ValueError("You can't define spacing and gap at the same time")
    if spacing is None:
        spacing = (size[0] + gap[0], size[1] + gap[1])

    c = Component()
    c.info["xspacing"], c.info["yspacing"] = spacing
    c.info["enclosure"] = enclosure
    c.info["xsize"], c.info["ysize"] = size
    c.info["size"] = tuple(size)
    c.info["spacing"] = tuple(spacing)

    width, height = size
    a = width / 2
    b = height / 2
    c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    bbox_layers = bbox_layers or []

    if bbox_offset:
        warnings.warn(
            "bbox_offset is deprecated. Use bbox_offsets instead", DeprecationWarning
        )
        bbox_offsets = [bbox_offset] * len(bbox_layers)

    bbox_offsets = bbox_offsets or []

    for layer, bbox_offset in zip(bbox_layers, bbox_layers):
        a = (width + bbox_offset) / 2
        b = (height + bbox_offset) / 2
        c.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    c.add_port("e1", center=(0, 0), width=height, orientation=0, layer=layer)
    return c


viac = partial(via, layer="VIAC")
via1 = partial(via, layer="VIA1", enclosure=2)
via2 = partial(via, layer="VIA2")


if __name__ == "__main__":
    c = via()
    # c.pprint()
    print(c)
    c.show(show_ports=True)
