from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs, Size


@gf.cell
def via_stack_with_offset(
    layers: LayerSpecs = ("PPP", "M1"),
    size: Size | None = (10, 10),
    sizes: Sequence[Size] | None = None,
    layer_offsets: Sequence[float] | None = None,
    vias: Sequence[ComponentSpec | None] = (None, "viac"),
    offsets: Sequence[float] | None = None,
    layer_to_port_orientations: dict[LayerSpec, list[int]] | None = None,
) -> Component:
    """Rectangular layer transition with offset between layers.

    Args:
        layers: layer specs between vias.
        size: for all vias array.
        sizes: Optional size for each via array. Overrides size.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: via spec for previous layer. None for no via.
        offsets: optional offset for each layer relatively to the previous one.
            By default it only offsets by size[1] if there is a via.
        layer_to_port_orientations: Optional dictionary with layer to port orientations.

    .. code::

        side view

         __________________________
        |                          |
        |                          | layers[2]
        |__________________________|           vias[2] = None
        |                          |
        | layer_offsets[1]+size    | layers[1]
        |__________________________|
            |     |
            vias[1]
         ___|_____|__
        |            |
        |  sizes[0]  |  layers[0]
        |____________|

            vias[0] = None

    """
    c = Component()
    y0 = 0.0

    if sizes and layer_offsets:
        raise ValueError("You need to set either sizes or layer_offsets")

    if size and sizes:
        raise ValueError("You need to set either size or sizes")

    offsets = list(offsets or [0] * len(layers))
    layer_offsets = list(layer_offsets or [0] * len(layers))
    if sizes:
        sizes_list = list(sizes)
    else:
        assert size is not None
        sizes_list = [size] * len(layers)

    elements = {len(layers), len(layer_offsets), len(vias), len(sizes_list)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias, {len(sizes_list)} sizes",
            stacklevel=3,
        )

    port_orientations = (180, 90, 0, -90)
    layer_to_port_orientations_dict = layer_to_port_orientations or {
        layers[-1]: list(port_orientations)
    }

    previous_layer = layers[0]

    for layer in layer_to_port_orientations_dict:
        if layer not in layers:
            raise ValueError(
                f"layer {layer} in layer_to_port_orientations not in layers {layers}"
            )

    for layer, via, size, size_offset, offset in zip(
        layers, vias, sizes_list, layer_offsets, offsets
    ):
        width, height = size
        width += 2 * size_offset
        height += 2 * size_offset
        x0 = -width / 2
        ref_layer = c << gf.c.compass(size=(width, height), layer=layer, port_type=None)
        ref_layer.dymin = y0

        if layer in layer_to_port_orientations_dict:
            ref_layer = c << gf.c.compass(
                size=(width, height),
                layer=layer,
                port_type="electrical",
                port_orientations=layer_to_port_orientations_dict[layer],
                auto_rename_ports=False,
            )
            ref_layer.ymin = int(y0)
            c.add_ports(ref_layer.ports)
        else:
            ref_layer = c << gf.c.compass(
                size=(width, height),
                layer=previous_layer,
                port_type=None,
                port_orientations=None,
            )
            ref_layer.ymin = int(y0)

        if via:
            via = gf.get_component(via)
            if "xsize" not in via.info:
                raise ValueError(f"via {via.name!r} is missing xsize info")
            if "ysize" not in via.info:
                raise ValueError(f"via {via.name!r} is missing ysize info")
            if "enclosure" not in via.info:
                raise ValueError(f"via {via.name!r} is missing enclosure info")
            if "column_pitch" not in via.info:
                raise ValueError(
                    f"Component {via.name!r} does not have a 'column_pitch' key in info"
                )
            if "row_pitch" not in via.info:
                raise ValueError(
                    f"Component {via.name!r} does not have a 'row_pitch' key in info"
                )

            w, h = via.info["xsize"], via.info["ysize"]
            enclosure = via.info["enclosure"]
            pitch_x = via.info["column_pitch"]
            pitch_y = via.info["row_pitch"]

            nb_vias_x = (width - w - 2 * enclosure) / pitch_x + 1
            nb_vias_y = (height - h - 2 * enclosure) / pitch_y + 1

            nb_vias_x = int(abs(floor(nb_vias_x))) or 1
            nb_vias_y = int(abs(floor(nb_vias_y))) or 1

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2

            x00 = x0 + cw + w / 2
            y00 = y0 + ch + h / 2 + offset

            ref = c.add_ref(
                via,
                columns=nb_vias_x,
                rows=nb_vias_y,
                column_pitch=pitch_x,
                row_pitch=pitch_y,
            )
            ref.dmove((x00, y00))
            y0 += height
            if ref.dxsize + enclosure > width or ref.dysize + enclosure > height:
                warnings.warn(
                    f"size = {size} for layer {layer} violates min enclosure"
                    f" {enclosure} for via {via.name!r}",
                    stacklevel=3,
                )

        y0 += offset
        previous_layer = layer

    ref = c << gf.c.compass(
        size=(width, height),
        layer=layers[-2],
        port_type=None,
        port_orientations=None,
    )
    ref.dymin = ref_layer.dymin
    return c


via_stack_with_offset_ppp_m1 = partial(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, "viac"),
)

via_stack_with_offset_ppp_m1 = partial(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, "viac"),
)

via_stack_with_offset_m1_m3 = partial(
    via_stack_with_offset,
    layers=("M1", "M2", "MTOP"),
    vias=(None, "via1", "via2"),
)


if __name__ == "__main__":
    c = via_stack_with_offset(
        layers=("M1", "M2", "MTOP"),
        size=None,
        sizes=((10, 10), (5, 5), (5, 5)),
        vias=(None, "via1", "via2"),
        # layer_to_port_orientations={"MTOP": [90], "M1": [270]},
    )
    # c = via_stack_with_offset_m1_m3(layer_offsets=[0, 5, 10])
    # c = via_stack_with_offset(vias=(None, None))
    c.show()
