from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, Floats, Ints, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name
def via_stack(
    size: Size = (11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | tuple[float | tuple[float, float], ...] | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_to_port_orientations: dict[LayerSpec, list[int]] | None = None,
    correct_size: bool = True,
    slot_horizontal: bool = False,
    slot_vertical: bool = False,
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    http://www.vlsi-expert.com/2017/12/vias.html

    Args:
        size: of the layers.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size. If a tuple, it is the offset in x and y.
        vias: vias to use to fill the rectangles.
        layer_to_port_orientations: dictionary of layer to port_orientations.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
        slot_horizontal: if True, then vias are horizontal.
        slot_vertical: if True, then vias are vertical.
        port_orientations: list of port_orientations to add. None does not add ports.
    """
    width_m, height_m = size
    a = width_m / 2
    b = height_m / 2

    layers = layers or []
    layer_indices = [gf.get_layer(layer) for layer in layers]
    layer_offsets = layer_offsets or [0] * len(layers)
    layer_to_port_orientations_list = layer_to_port_orientations or {
        gf.get_layer(layers[-1]): list(port_orientations or [])
    }

    elements = {len(layers), len(layer_offsets), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    c = Component()
    c.info["xsize"], c.info["ysize"] = size

    for layer_index, offset in zip(layer_indices, layer_offsets):
        if isinstance(offset, Iterable):
            offset_x = offset[0]
            offset_y = offset[1]
        else:
            offset_x = offset_y = offset

        size_m = (width_m + 2 * offset_x, height_m + 2 * offset_y)

        if layer_index in layer_to_port_orientations_list:
            ref = c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type="electrical",
                port_orientations=layer_to_port_orientations_list[layer_index],
                auto_rename_ports=False,
            )
            c.add_ports(ref.ports)
        else:
            ref = c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type=None,
                port_orientations=port_orientations,
            )
        # c.absorb(ref)

    vias_list = vias or []
    for via, offset in zip(vias_list, layer_offsets):
        if via is not None:
            width, height = size
            if isinstance(offset, Iterable):
                offset_x = offset[0]
                offset_y = offset[1]
            else:
                offset_x = offset_y = offset
            width += 2 * offset_x
            height += 2 * offset_y
            _via = gf.get_component(via)

            if "xsize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'xsize' key in info"
                )
            if "ysize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'ysize' key in info"
                )

            if "column_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'column_pitch' key in info"
                )
            if "row_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'row_pitch' key in info"
                )

            w, h = _via.xsize, _via.ysize
            enclosure = _via.info["enclosure"]
            pitch_y = _via.info["row_pitch"]
            pitch_x = _via.info["column_pitch"]

            min_width = w + enclosure
            min_height = h + enclosure

            if slot_horizontal:
                width = size[0] - 2 * enclosure
                via = gf.get_component(via, size=(width, h))
                nb_vias_x = 1
                nb_vias_y = abs(height - h - 2 * enclosure) / pitch_y + 1

            elif slot_vertical:
                height = size[1] - 2 * enclosure
                via = gf.get_component(via, size=(w, height))
                nb_vias_x = abs(width - w - 2 * enclosure) / pitch_x + 1
                nb_vias_y = 1
            else:
                via = _via
                nb_vias_x = abs(width - w - 2 * enclosure) / pitch_x + 1
                nb_vias_y = abs(height - h - 2 * enclosure) / pitch_y + 1

            min_width = w + enclosure
            min_height = h + enclosure

            if (min_width > width and correct_size) or (
                min_width <= width and min_height > height and correct_size
            ):
                warnings.warn(
                    f"Changing size from ({width}, {height}) to ({min_width}, {min_height}) to fit a via!",
                    stacklevel=3,
                )
                width = max(min_width, width)
                height = max(min_height, height)
            elif min_width > width or min_height > height:
                raise ValueError(f"size {size} is too small to fit a {(w, h)} um via")

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1
            ref = c.add_ref(
                via,
                columns=nb_vias_x,
                rows=nb_vias_y,
                column_pitch=pitch_x,
                row_pitch=pitch_y,
            )

            a = width / 2
            b = height / 2
            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))
    return c


@gf.cell_with_module_name
def via_stack_corner45(
    width: float = 10,
    layers: Sequence[LayerSpec | None] = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_port: LayerSpec | None = None,
    correct_size: bool = True,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        width: of the corner45.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
    """
    height = width
    layers_list = layers or []
    layer_offsets_list = layer_offsets or [0] * len(layers_list)

    elements = {len(layers_list), len(layer_offsets_list), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers_list)} layers, {len(layer_offsets_list)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    if layers_list:
        layer_port = layer_port or layers_list[-1]

    c = Component()
    if layer_port:
        c.info["layer"] = layer_port

    ref: ComponentReference | None = None
    for layer, offset in zip(layers_list, layer_offsets_list):
        if layer and layer == layer_port:
            ref = c << gf.c.wire_corner45(
                width=width + 2 * offset, layer=layer, with_corner90_ports=False
            )
            c.add_ports(ref.ports)
        elif layer is not None:
            ref = c << gf.c.wire_corner45(
                width=width + 2 * offset, layer=layer, with_corner90_ports=False
            )
    assert ref is not None

    width_corner = width
    width = ref.xsize
    height = ref.ysize
    xmin = ref.xmin
    ymin = ref.ymin

    vias_list = vias or []
    for via, offset in zip(vias_list, layer_offsets_list):
        if via is not None:
            width45 = (
                2 * (width_corner + 2 * offset) * np.cos(np.deg2rad(45))
            )  # Width in the x direction
            _via = gf.get_component(via)
            if "xsize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'xsize' key in info"
                )
            if "ysize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'ysize' key in info"
                )

            if "column_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'column_pitch' key in info"
                )
            if "row_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'row_pitch' key in info"
                )

            w, h = _via.info["xsize"], _via.info["ysize"]
            enclosure = _via.info["enclosure"]
            pitch_x = _via.info["column_pitch"]
            pitch_y = _via.info["row_pitch"]

            via = _via

            min_width = w + enclosure
            min_height = h + enclosure

            if (min_width > width45 and correct_size) or (
                min_width <= width45 and min_height > height and correct_size
            ):
                warnings.warn(
                    f"Changing size from ({width}, {height}) to ({min_width}, {min_height}) to fit a via!",
                    stacklevel=3,
                )
                width45 = max(min_width, width45)
                height = max(min_height, height)
            elif min_width > width45 or min_height > height:
                raise ValueError(
                    f"{min_width=} > {width=} or {min_height=} > {height=}"
                )

            # Keep placing rows until we cover the whole height
            y_covered = enclosure

            while y_covered + enclosure < height:
                y = ymin + y_covered + h / 2  # Position of the via

                # x offset from the edge of the metal to make sure enclosure is fulfilled
                xoff_enc = 2 * enclosure * np.cos(np.deg2rad(45))
                xoff = (y_covered + h) * np.tan(np.deg2rad(45)) + xoff_enc

                xpos0 = xmin + xoff

                # Calculate the number of vias that fit in a given width
                if (y_covered + h) < (height - width45):
                    # The x width is width45
                    xwidth = width45
                else:
                    # The x width is decreasing
                    xwidth = (height - (y_covered + h)) * np.tan(np.deg2rad(45))

                if min_width <= xwidth:
                    vias_per_row = (
                        xwidth - 2 * xoff_enc - 2 * h * np.tan(np.deg2rad(45))
                    ) / (pitch_x) + 1
                    # Place the vias at the given x, y
                    for i in range(int(vias_per_row)):
                        ref = c << via
                        ref.center = (xpos0 + pitch_x * i + w / 2, y)

                y_covered = y_covered + h + pitch_y

    c.flatten()
    return c


@gf.cell_with_module_name
def via_stack_corner45_extended(
    corner: ComponentSpec = "via_stack_corner45",
    via_stack: ComponentSpec = "via_stack",
    width: float = 3,
    length: float = 10,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        corner: corner component.
        via_stack: for the via stack.
        width: of the corner45.
        length: of the straight.
    """
    c = gf.Component()
    corner_component = c << gf.get_component(corner, width=width / np.sqrt(2))
    s = gf.get_component(via_stack, size=(length, width))
    sr = c << s
    sl = c << s
    sr.connect("e1", corner_component.ports["e1"])
    sl.connect("e1", corner_component.ports["e2"])
    return c


via_stack_m1_mtop = via_stack_m1_m3 = partial(
    via_stack,
    layers=("M1", "M2", "MTOP"),
    vias=("via1", "via2", None),
)
via_stack_m2_m3 = partial(
    via_stack,
    layers=("M2", "MTOP"),
    vias=("via2", None),
)
via_stack_slab_m1 = partial(
    via_stack,
    layers=("SLAB90", "M1"),
    vias=("viac", "via1"),
)
via_stack_slab_m2 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2"),
    vias=("viac", "via1", None),
)

via_stack_slab_m3 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "MTOP"),
    vias=("viac", "via1", "via2", None),
)
via_stack_npp_m1 = partial(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_slab_npp_m3 = partial(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_heater_mtop = via_stack_heater_m3 = partial(
    via_stack, layers=("HEATER", "M2", "MTOP"), vias=(None, "via1", "via2")
)
via_stack_heater_mtop_mini = partial(via_stack_heater_mtop, size=(4, 4))

via_stack_heater_m2 = partial(via_stack, layers=("HEATER", "M2"), vias=(None, "via1"))

via_stack_slab_m1_horizontal = partial(via_stack_slab_m1, slot_horizontal=True)


if __name__ == "__main__":
    c = via_stack_heater_mtop()
    c.pprint_ports()
    c.show()
