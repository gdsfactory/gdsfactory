from __future__ import annotations

import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, via2, viac
from gdsfactory.typings import ComponentSpec, Float2, Floats, LayerSpec, LayerSpecs


@gf.cell
def via_stack(
    size=(11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    layer_offsets: Optional[Floats] = None,
    vias: Optional[Tuple[Optional[ComponentSpec], ...]] = (via1, via2, None),
    layer_port: Optional[LayerSpec] = None,
    correct_size: bool = True,
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
    c.info["size"] = (float(size[0]), float(size[1]))
    c.info["layer"] = layer_port

    for layer, offset in zip(layers, layer_offsets):
        size_m = (width_m + 2 * offset, height_m + 2 * offset)
        if layer == layer_port:
            ref = c << compass(size=size_m, layer=layer, port_type="electrical")
            c.add_ports(ref.ports)
        else:
            ref = c << compass(size=size_m, layer=layer, port_type="electrical")

    vias = vias or []
    for via, offset in zip(vias, layer_offsets):
        if via is not None:
            width, height = size
            width += 2 * offset
            height += 2 * offset
            via = gf.get_component(via)
            w, h = via.info["size"]
            enclosure = via.info["enclosure"]
            pitch_x, pitch_y = via.info["spacing"]

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

            nb_vias_x = abs(width - w - 2 * enclosure) / pitch_x + 1
            nb_vias_y = abs(height - h - 2 * enclosure) / pitch_y + 1

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
def via_stack_circular(
    radius: float = 10.0,
    angular_extent: float = 45,
    center_angle: float = 0,
    width: float = 5.0,
    layers: LayerSpecs = ("M1", "M2", "M3"),
    vias: Tuple[Optional[ComponentSpec], ...] = (via1, via2),
    layer_port: Optional[LayerSpec] = None,
) -> Component:
    """Circular via array stack.

    FIXME! does not work.

    Constructs a circular via array stack. It does so
    by stacking rectangular via stacks offset by a small amount
    along the specified circumference.

    Args:
        radius: of the via stack (center).
        angular_extent: of the via stack.
        center_angle: of the via stack.
        width: of the via stack.
        layers: layers to draw
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
    """

    # We basically just want to place rectangular via stacks
    # stacked with a little bit of an offset
    c = gf.Component()

    layers = layers or []

    if layers:
        layer_port = layer_port or layers[-1]

    init_angle = (center_angle - angular_extent / 2) * np.pi / 180
    end_angle = (center_angle + angular_extent / 2) * np.pi / 180

    init_angle = init_angle % (2 * np.pi)
    if init_angle > np.pi:
        init_angle = init_angle - 2 * np.pi

    end_angle = end_angle % (2 * np.pi)
    if end_angle > np.pi:
        end_angle = end_angle - 2 * np.pi

    # We do this via-centric: we figure out the min spacing between vias,
    # and from that figure out all the metal dimensions
    # This will of course fail if no via information is provided,
    # but why would you instantiate a ViaStack without any via?

    for level, via in enumerate(vias):
        if via is None:
            continue

        metal_bottom = layers[level]
        metal_top = layers[level + 1]
        via = gf.get_component(via)

        w, h = via.info["size"]
        g = via.info["enclosure"]
        pitch_x, pitch_y = via.info["spacing"]

        nb_vias_x = (width - w - 2 * g) / pitch_x + 1
        nb_vias_x = int(np.floor(nb_vias_x)) or 1

        size_metal = (width, h + 2 * g)

        # Now start placing via lines at each angle starting from
        # the initial angle until we reach the end angle
        ang = init_angle

        while _smaller_angle(ang, ang, end_angle):
            pos = radius * np.array((np.cos(ang), np.sin(ang)))

            ref = c.add_array(
                via, columns=nb_vias_x, rows=1, spacing=(pitch_x, pitch_y)
            )
            ref.center = pos

            # Place top and bottom metal
            for metal in [metal_top, metal_bottom]:
                met = c << gf.components.rectangle(size=size_metal, layer=metal)
                met.center = pos

            # Let's see if we can do something different
            x, y = pos

            if x > 0:
                new_y = y + pitch_y
                mult = 1
            else:
                new_y = y - pitch_y
                mult = -1

            if new_y > radius:
                new_y = y - pitch_y
                assert new_y < radius
                new_x = -1 * np.sqrt(np.power(radius, 2) - np.power(new_y, 2))
            elif new_y < -radius:
                new_y = y + pitch_y
                assert new_y > -radius
                new_x = np.sqrt(np.power(radius, 2) - np.power(new_y, 2))

            else:
                new_x = mult * np.sqrt(np.power(radius, 2) - np.power(new_y, 2))

            if np.isnan(new_x):
                print(radius)
                print(new_y)
                print(np.power(radius, 2) - np.power(new_y, 2))
            assert not np.isnan(new_x)
            ang = np.arctan2(new_y, new_x)

    return c


def _smaller_angle(angle, angle1, angle2):
    """Returns False if angle is outside the
     bounds of the arc angle defined between
     angle 1 and angle2.

    But it does so assuming that angle1 and angle2 are between [-pi, pi]
    and that we are trying to fill an arc
    """

    if angle2 >= 0 and angle1 >= 0:
        if angle2 > angle1:
            return angle < angle2
        # Convert angle to 0, 2pi and see if out of bounds
        angle = angle + 2 * np.pi * (angle < 0)
        return not (angle2 < angle < angle1)

    elif angle2 < 0 and angle1 < 0:
        return angle < angle2 if angle2 > angle1 else not (angle2 < angle < angle1)
    else:
        if angle2 < 0 and angle > 0 or angle2 >= 0 and angle < 0:
            return True
        else:
            return angle < angle2


@gf.cell
def via_stack_from_rules(
    size: Float2 = (1.2, 1.2),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    layer_offsets: Optional[Tuple[float, ...]] = None,
    vias: Optional[Tuple[Optional[ComponentSpec], ...]] = (via1, via2),
    via_min_size: Tuple[Float2, ...] = ((0.2, 0.2), (0.2, 0.2)),
    via_min_gap: Tuple[Float2, ...] = ((0.1, 0.1), (0.1, 0.1)),
    via_min_enclosure: Float2 = (0.15, 0.25),
    layer_port: Optional[LayerSpec] = None,
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
    c.info["size"] = (float(size[0]), float(size[1]))
    c.info["layer"] = layer_port

    layer_offsets = layer_offsets or [0] * len(layers)

    for layer, offset in zip(layers, layer_offsets):
        size = (width + 2 * offset, height + 2 * offset)
        if layer == layer_port:
            ref = c << compass(size=size, layer=layer, port_type="electrical")
            c.add_ports(ref.ports)
        else:
            ref = c << compass(size=size, layer=layer, port_type="electrical")

    vias = vias or []
    c.info["vias"] = []
    for current_via, min_size, min_gap, min_enclosure in zip(
        vias, via_min_size, via_min_gap, via_min_enclosure
    ):
        if current_via is not None:
            # Optimize via
            via = gf.get_component(
                optimized_via(current_via, size, min_size, min_gap, min_enclosure)
            )
            c.info["vias"].append(via.info)

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
    size: Tuple[float, float] = (11.0, 11.0),
    min_via_size: Tuple[float, float] = (0.3, 0.3),
    min_via_gap: Tuple[float, float] = (0.1, 0.1),
    min_via_enclosure: float = 0.2,
) -> Component:
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


def test_via_stack_from_rules() -> None:
    # Check that vias are generated with larger than min dimensions if possible
    size = (1.2, 1.2)
    layers = ("M1", "M2", "M3")
    vias = (via1, via2)
    via_min_size = ((0.2, 0.2), (0.2, 0.2))
    via_min_gap = ((0.1, 0.1), (0.15, 0.15))
    via_min_enclosure = (0.1, 0.1)

    c = gf.get_component(
        via_stack_from_rules(
            size=size,
            layers=layers,
            vias=vias,
            via_min_size=via_min_size,
            via_min_gap=via_min_gap,
            via_min_enclosure=via_min_enclosure,
        )
    )

    assert c.info["vias"][0]["size"][0] > via_min_size[0][0]
    assert c.info["vias"][0]["size"][1] > via_min_size[0][1]
    assert (
        c.info["vias"][0]["spacing"][0]
        == via_min_gap[0][0] + c.info["vias"][0]["size"][0]
    )


via_stack_m1_m3 = partial(
    via_stack,
    layers=("M1", "M2", "M3"),
    vias=(via1, via2, None),
)

via_stack_slab_m3 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "M3"),
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
via_stack_heater_mtop = via_stack_heater_m3 = partial(
    via_stack, layers=("HEATER", "M2", "M3"), vias=(None, via1, via2)
)


if __name__ == "__main__":
    c = via_stack()
    c.show()
    # c = gf.pack([via_stack_slab_m3, via_stack_heater_mtop])[0]

    # c = gf.Component("offgrid_demo")
    # v1 = c << via_stack_slab_m3()
    # v2 = c << via_stack_slab_m3()
    # v2.x = 20.0005
    # c.show()

    # c2 = gf.Component()
    # c21 = c2 << c
    # c22 = c2 << c
    # c22.x = 20.0005 + 30
    # c2.show()

    # c = via_stack_heater_mtop(layer_offsets=(0, 1, 2))
    # c = via_stack_circular()
    # c = via_stack_m1_m3(size=(4.5, 4.5))
    # print(c.to_dict())
    # c.show(show_ports=True)

    # c = via_stack_from_rules()
    # c.show(show_ports=True)

    # c = via_stack_circular(
    #     radius=20.0,
    #     angular_extent=300,
    #     center_angle=0,
    #     width=5.0,
    #     layers=("M1", "M2", "M3"),
    #     vias=(via1, via2),
    #     layer_port=None,
    # )
    # c.show(show_ports=True)
