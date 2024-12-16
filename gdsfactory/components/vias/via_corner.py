from __future__ import annotations

from typing import Any

from numpy import floor

import gdsfactory as gf
from gdsfactory.cross_section import metal2, metal3
from gdsfactory.port import select_ports
from gdsfactory.typings import ComponentSpec, MultiCrossSectionAngleSpec


@gf.cell
def via_corner(
    cross_section: MultiCrossSectionAngleSpec = (
        (metal2, (0, 180)),
        (metal3, (90, 270)),
    ),
    vias: tuple[ComponentSpec] = ("via1",),
    layers_labels: tuple[str, ...] = ("m2", "m3"),
    **kwargs: Any,
) -> gf.Component:
    """Returns Corner via.

    Use in place of wire_corner to route between two layers.

    Args:
        cross_section: list of cross_section, orientation pairs.
        vias: vias to use to fill the rectangles.
        layers_labels: Labels to use for each layer.
        kwargs: cross_section settings.
    """
    cross_sections = [gf.get_cross_section(x[0], **kwargs) for x in cross_section]
    port_orientations = [x[1] for x in cross_section]
    widths = heights = [x.width for x in cross_sections]
    layers = [x.layer for x in cross_sections]
    layers_ports = layers

    max_width = max(widths)
    max_height = max(heights)
    min_height = min(heights)
    min_width = min(widths)

    a = min_width / 2
    b = min_height / 2

    c = gf.Component()
    c.info["size"] = (float(max_width), float(max_height))
    c.info["length"] = max(max_width, max_height)

    port_type = "electrical"
    for i, layer in enumerate(layers):
        assert layer is not None
        ref = c << gf.c.compass(
            size=(widths[i], heights[i]), layer=layer, port_type=port_type
        )
        if layer in layers_ports:
            orientations = port_orientations[i]
            if (90 in orientations) or (270 in orientations):
                orientation = 90
            elif (0 in orientations) or (180 in orientations):
                orientation = 180
            else:
                raise ValueError(f"Port orientation {orientations} not valid.")
            ports = select_ports(ref.ports, orientation=orientation)
            c.add_ports(ports, prefix=f"{layers_labels[i]}_")

    for via in vias:
        via = gf.get_component(via)
        if "xsize" not in via.info:
            raise ValueError(f"Via {via.name!r} does not have xsize in info")

        if "ysize" not in via.info:
            raise ValueError(f"Via {via.name!r} does not have ysize in info")

        if "enclosure" not in via.info:
            raise ValueError(f"Via {via.name!r} does not have enclosure in info")

        if "column_pitch" not in via.info:
            raise ValueError(
                f"Component {via.name!r} does not have a 'column_pitch' key in info"
            )
        if "row_pitch" not in via.info:
            raise ValueError(
                f"Component {via.name!r} does not have a 'row_pitch' key in info"
            )

        w = via.info["xsize"]
        h = via.info["ysize"]
        g = via.info["enclosure"]
        pitch_y = via.info["row_pitch"]
        pitch_x = via.info["column_pitch"]

        nb_vias_x = (min_width - w - 2 * g) / pitch_x + 1
        nb_vias_y = (min_height - h - 2 * g) / pitch_y + 1

        nb_vias_x = int(floor(nb_vias_x)) or 1
        nb_vias_y = int(floor(nb_vias_y)) or 1
        ref = c.add_ref(
            via,
            columns=nb_vias_x,
            rows=nb_vias_y,
            column_pitch=pitch_x,
            row_pitch=pitch_y,
        )

        cw = (min_width - (nb_vias_x - 1) * pitch_x - w) / 2
        ch = (min_height - (nb_vias_y - 1) * pitch_y - h) / 2
        x0 = -a + cw + w / 2
        y0 = -b + ch + h / 2
        ref.dmove((x0, y0))
    return c


if __name__ == "__main__":
    c = via_corner()
    c.show()
