from __future__ import annotations

import warnings
from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, via2, viac
from gdsfactory.components.wire import wire_corner45
from gdsfactory.typings import (
    ComponentFactory,
    ComponentSpec,
    Float2,
    Floats,
    LayerSpec,
    LayerSpecs,
)


@gf.cell
def via_stack(
    size=(11.0, 11.0),
    layers: tuple[LayerSpec | None, ...] = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | None = None,
    vias: tuple[ComponentSpec | None, ...] | None = (via1, via2, None),
    layer_port: LayerSpec | None = None,
    correct_size: bool = True,
    slot_horizontal: bool = False,
    slot_vertical: bool = False,
) -> Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    http://www.vlsi-expert.com/2017/12/vias.html

    spacing = via.info['spacing']
    enclosure = via.info['enclosure']

    Args:
        size: of the layers.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
        slot_horizontal: if True, then vias are horizontal.
        slot_vertical: if True, then vias are vertical.
    """
    width_m, height_m = size
    a = width_m / 2
    b = height_m / 2

    layers = layers or []
    layer_offsets = layer_offsets or [0] * len(layers)

    elements = {len(layers), len(layer_offsets), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    if layers:
        layer_port = layer_port or layers[-1]

    c = Component()
    c.height = height_m
    c.info["size"] = tuple(size)
    c.info["xsize"] = size[0]
    c.info["ysize"] = size[1]
    if layer_port:
        c.info["layer"] = layer_port

    for layer, offset in zip(layers, layer_offsets):
        size_m = (width_m + 2 * offset, height_m + 2 * offset)
        if layer and layer == layer_port:
            ref = c << compass(size=size_m, layer=layer, port_type="electrical")
            c.add_ports(ref.ports)
        elif layer is not None:
            ref = c << compass(size=size_m, layer=layer, port_type="electrical")

    vias = vias or []
    for via, offset in zip(vias, layer_offsets):
        if via is not None:
            width, height = size
            width += 2 * offset
            height += 2 * offset
            _via = gf.get_component(via)
            w, h = _via.info["xsize"], _via.info["ysize"]
            enclosure = _via.info["enclosure"]
            pitch_x, pitch_y = _via.info["xspacing"], _via.info["yspacing"]

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

            if (
                min_width > width
                and correct_size
                or min_width <= width
                and min_height > height
                and correct_size
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
            ref = c.add_array(
                via, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )
            if ref.xsize + enclosure > width or ref.ysize + enclosure > height:
                warnings.warn(
                    f"size = {size} for layer {layer} violates min enclosure"
                    f" {enclosure} for via {via.name!r}",
                    stacklevel=3,
                )

            a = width / 2
            b = height / 2
            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))

    return c


@gf.cell
def via_stack_corner45(
    width: float = 10,
    layers: tuple[LayerSpec | None, ...] = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | None = None,
    vias: tuple[ComponentSpec | None, ...] | None = (via1, via2, None),
    layer_port: LayerSpec | None = None,
    correct_size: bool = True,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    spacing = via.info['spacing']
    enclosure = via.info['enclosure']

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
    layers = layers or []
    layer_offsets = layer_offsets or [0] * len(layers)

    elements = {len(layers), len(layer_offsets), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    if layers:
        layer_port = layer_port or layers[-1]

    c = Component()
    if layer_port:
        c.info["layer"] = layer_port

    for layer in layers:
        if layer and layer == layer_port:
            ref = c << wire_corner45(
                width=width, layer=layer, with_corner90_ports=False
            )
            c.add_ports(ref.ports)
        elif layer is not None:
            ref = c << wire_corner45(
                width=width, layer=layer, with_corner90_ports=False
            )

    width_corner = width
    width, height = ref.size
    xmin = ref.xmin
    ymin = ref.ymin

    vias = vias or []
    for via, offset in zip(vias, layer_offsets):
        if via is not None:
            width += 2 * offset
            height += 2 * offset
            _via = gf.get_component(via)
            w, h = _via.info["xsize"], _via.info["ysize"]
            enclosure = _via.info["enclosure"]
            pitch_x, pitch_y = _via.info["xspacing"], _via.info["yspacing"]

            via = _via
            nb_vias_x = abs(width - w - 2 * enclosure) / pitch_x + 1
            nb_vias_y = abs(height - h - 2 * enclosure) / pitch_y + 1

            min_width = w + enclosure
            min_height = h + enclosure

            if (
                min_width > width
                and correct_size
                or min_width <= width
                and min_height > height
                and correct_size
            ):
                warnings.warn(
                    f"Changing size from ({width}, {height}) to ({min_width}, {min_height}) to fit a via!",
                    stacklevel=3,
                )
                width = max(min_width, width)
                height = max(min_height, height)
            elif min_width > width or min_height > height:
                raise ValueError(
                    f"{min_width=} > {width=} or {min_height=} > {height=}"
                )

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1

            nrows = (width_corner - 2 * enclosure) / pitch_x + 1

            for i in range(nb_vias_x):
                for j in range(nb_vias_y):
                    x, y = (
                        xmin + enclosure + i * pitch_x,
                        ymin + enclosure + j * pitch_y,
                    )

                    for row in range(1, int(nrows) + 1):
                        if i - row == j:
                            ref = c << via
                            ref.center = (x, y)
    return c


@gf.cell
def via_stack_corner45_extended(
    corner: ComponentSpec = via_stack_corner45,
    via_stack: ComponentSpec = via_stack,
    width: float = 3,
    length: float = 10,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        corner: corner component.
        straight: straight component.
        width: of the corner45.
        length: of the straight.
    """
    c = gf.Component()
    corner = c << gf.get_component(corner, width=width / np.sqrt(2))
    s = gf.get_component(via_stack, size=(length, width))
    sr = c << s
    sl = c << s
    sr.connect("e1", corner.ports["e1"])
    sl.connect("e1", corner.ports["e2"])
    return c


@gf.cell
def via_stack_from_rules(
    size: Float2 = (1.2, 1.2),
    layers: LayerSpecs = ("M1", "M2", "MTOP"),
    layer_offsets: tuple[float, ...] | None = None,
    vias: tuple[ComponentSpec | None, ...] | None = (via1, via2),
    via_min_size: tuple[Float2, ...] = ((0.2, 0.2), (0.2, 0.2)),
    via_min_gap: tuple[Float2, ...] = ((0.1, 0.1), (0.1, 0.1)),
    via_min_enclosure: Float2 = (0.15, 0.25),
    layer_port: LayerSpec | None = None,
) -> Component:
    """Rectangular via array stack, with optimized dimension for vias.

    Uses inclusion, minimum width, and minimum spacing rules to place the maximum number of individual vias,
    each with maximum via area.

    Args:
        size: of the layers, len(size).
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: list of base via components to modify.
        via_min_size: via minimum x, y dimensions.
        via_min_gap: via minimum x, y distances.
        via_min_enclosure: via minimum inclusion into connecting layers.
        layer_port: if None assumes port is on the last layer.
    """
    width, height = size
    a = width / 2
    b = height / 2

    layers = layers or []

    if layers:
        layer_port = layer_port or layers[-1]

    c = Component()
    c.height = height
    c.info["xsize"] = size[0]
    c.info["ysize"] = size[1]
    c.info["layer"] = layer_port

    layer_offsets = layer_offsets or [0] * len(layers)

    for layer, offset in zip(layers, layer_offsets):
        size = (width + 2 * offset, height + 2 * offset)
        if layer and layer == layer_port:
            ref = c << compass(size=size, layer=layer, port_type="electrical")
            c.add_ports(ref.ports)
        elif layer:
            ref = c << compass(size=size, layer=layer, port_type="electrical")

    vias = vias or []
    for current_via, min_size, min_gap, min_enclosure in zip(
        vias, via_min_size, via_min_gap, via_min_enclosure
    ):
        if current_via is not None:
            # Optimize via
            via = gf.get_component(
                optimized_via(current_via, size, min_size, min_gap, min_enclosure)
            )

            w, h = via.info["size"]
            g = via.info["enclosure"]
            pitch_x, pitch_y = via.info["spacing"]

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1
            ref = c.add_array(
                via, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))

    return c


def optimized_via(
    base_via: ComponentSpec = "VIAC",
    size: tuple[float, float] = (11.0, 11.0),
    min_via_size: tuple[float, float] = (0.3, 0.3),
    min_via_gap: tuple[float, float] = (0.1, 0.1),
    min_via_enclosure: float = 0.2,
) -> ComponentFactory:
    """Given a target total inclusion size, returns an optimized dimension for the via.

    Uses inclusion, minimum width, and minimum spacing rules to place the maximum number of individual vias, with maximum via area.

    Arguments:
        base_via: to modify.
        size: of the target enclosing medium.
        min_via_size: minimum size the vias can take.
        min_via_gap: minimum distance between vias.
        min_via_enclosure: minimum distance between edge of enclosing medium and nearest via edge.
    """
    via_size = [0, 0]
    for dim in [0, 1]:
        via_area = size[dim] - 2 * min_via_enclosure + min_via_gap[dim]
        num_vias = int(via_area / (min_via_size[dim] + min_via_gap[dim]))
        try:
            via_size[dim] = float(via_area / num_vias) - min_via_gap[dim]
        except ZeroDivisionError as e:
            raise RuntimeError(
                "Cannot fit vias with specified minimum dimensions in provided space."
            ) from e

    return partial(
        base_via,
        size=via_size,
        gap=min_via_gap,
        spacing=None,
        enclosure=min_via_enclosure,
    )


via_stack_m1_m3 = partial(
    via_stack,
    layers=("M1", "M2", "MTOP"),
    vias=(via1, via2, None),
)

via_stack_slab_m3 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "MTOP"),
    vias=(viac, via1, via2, None),
)
via_stack_slab_m2 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2"),
    vias=(viac, via1, None),
)
via_stack_npp_m1 = partial(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_slab_npp_m3 = partial(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_heater_m2 = partial(via_stack, layers=("HEATER", "M2"), vias=(None, via1))
via_stack_heater_mtop = via_stack_heater_m3 = partial(
    via_stack, layers=("HEATER", "M2", "MTOP"), vias=(None, via1, via2)
)


if __name__ == "__main__":
    # c = via_stack()
    # c = gf.pack([via_stack_slab_m3, via_stack_heater_mtop])[0]
    # c = via_stack_slab_m3(size=(100, 10), slot_vertical=True)
    # c = via_stack_from_rules()
    # c = via_stack_corner45()
    c = via_stack_corner45_extended()
    c.show(show_ports=True)
