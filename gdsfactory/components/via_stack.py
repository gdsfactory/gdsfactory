from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, via2, viac
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs, Float2


@gf.cell
def via_stack(
    size=(11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    layer_offsets: Optional[Tuple[float, ...]] = None,
    vias: Optional[Tuple[Optional[ComponentSpec], ...]] = (via1, via2),
    layer_port: LayerSpec = None,
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
            ref = c << compass(size=size, layer=layer, port_type="placement")

    vias = vias or []
    for via_type, offs in zip(vias, layer_offsets):
        if via_type is not None:
            via_type = gf.get_component(via_type)

            w, h = via_type.info["size"]
            g = via_type.info["enclosure"]
            pitch_x, pitch_y = via_type.info["spacing"]

            min_width = w + g
            min_height = h + g

            if min_width > width or min_height > height:
                raise ValueError(f"size {size} is too small to fit a {(w, h)} um via")

            nb_vias_x = (width + 2 * offs - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height + 2 * offs - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1
            ref = c.add_array(
                via_type, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )

            cw = (width + 2 * offs - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height + 2 * offs - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a - offs + cw + w / 2
            y0 = -b - offs + ch + h / 2
            ref.move((x0, y0))

    return c


@gf.cell
def circular_via_stack(
    radius: float = 10.0,
    angular_extent: float = 45,
    center_angle: float = 0,
    width: float = 5.0,
    layers: LayerSpecs = ("M1", "M2", "M3"),
    vias: Tuple[Optional[ComponentSpec], ...] = (via1, via2),
    layer_port: LayerSpec = None,
) -> Component:
    """Circular via array stack.

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

    if angular_extent > 180:
        raise ValueError("This is not supported yet!")

    layers = layers or []

    if layers:
        layer_port = layer_port or layers[-1]

    h_center = radius * np.sin(center_angle * np.pi / 180)

    # Need to divide between 2 angular sections for it to work correctly
    init_angle = (center_angle - angular_extent / 2) % 360
    end_angle = (center_angle + angular_extent / 2) % 360

    if init_angle > end_angle:
        foo = end_angle
        end_angle = init_angle
        init_angle = foo

    if 0 < init_angle < 90 or 270 < init_angle < 360:
        if end_angle < 90:
            sections = [(init_angle, end_angle)]
        else:
            if end_angle < 270:
                sections = [(init_angle, 90.0), (90.0, end_angle)]
            else:
                sections = [(init_angle, 90.0), (90.0, 270.0), (270.0, end_angle)]

    else:
        if end_angle < 270.0:
            sections = [(init_angle, end_angle)]
        else:
            if end_angle < 90.0:
                sections = [(init_angle, 270.0), (270.0, end_angle)]
            else:
                sections = [(init_angle, 270.0), (270.0, 90.0), (90.0, end_angle)]

    c = Component()

    # We do this via-centric: we figure out the min spacing between vias,
    # and from that figure out all the metal dimensions
    # This will of course fail if no via information is provided,
    # but why would you instantiate a ViaStack without any via?

    print(sections)
    print("----")

    for sec in sections:
        cent_angle = (sec[0] + sec[1]) / 2
        ang_extent = sec[1] - sec[0]
        print(cent_angle)
        print(ang_extent)
        total_height = np.abs(
            radius
            * (
                np.sin((cent_angle + ang_extent / 2) * np.pi / 180)
                - np.sin((cent_angle - ang_extent / 2) * np.pi / 180)
            )
        )

        if 90 < cent_angle < 180 or 270 < cent_angle < 360:
            print("here")
            mult = -1
        else:
            mult = 0

        for level, via_type in enumerate(vias):
            metal_bottom = layers[level]
            metal_top = layers[level + 1]

            via_type = gf.get_component(via_type)

            # Get via info
            w, h = via_type.info["size"]
            g = via_type.info["enclosure"]
            pitch_x, pitch_y = via_type.info["spacing"]

            min_width = w + g
            min_height = h + g

            if min_width > width:
                raise ValueError(
                    f"The specified circular ViaStack width is too small to fit a via. Width is {width}um and the min is {min_width}um"
                )
            if min_height > total_height:
                raise ValueError(
                    f"The specified circular ViaStack height is too small to fit a via. Height is {total_height} um and the min is {min_height}um"
                )

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (total_height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1

            # Now place the via stack line by line so that we place the correct offset to
            # make it circular, as well as add the metal that encloses the via line

            h = h_center + (nb_vias_y - 1) * pitch_y / 2

            print(h)
            print(radius)
            print("===")

            angle = np.arcsin(h / (mult * radius))
            print(angle * 180 / np.pi)

            pos = (radius * np.cos(angle), radius * np.sin(angle))
            print(pos)

            size_metal = (width, h + 2 * g)

            for _ in range(nb_vias_y):
                # Place the via line
                ref = c.add_array(
                    via_type, columns=nb_vias_x, rows=1, spacing=(pitch_x, pitch_y)
                )
                ref.center = pos

                # Place top and bottom metal
                for metal in [metal_top, metal_bottom]:
                    met = c << gf.components.rectangle(size=size_metal, layer=metal)
                    met.center = pos

                h = h - pitch_y
                angle = np.arcsin(h / (mult * radius))
                pos = (radius * np.cos(angle), radius * np.sin(angle))

    # # Port placement
    # for layer in layers:
    #     size = (width + 2 * offset, height + 2 * offset)
    #     if layer == layer_port:
    #         ref = c << compass(size=size, layer=layer, port_type="electrical")
    #         c.add_ports(ref.ports)
    #     else:
    #         ref = c << compass(size=size, layer=layer, port_type="placement")

    return c


@gf.cell
def via_stack_from_rules(
    size: Float2 = (1.2, 1.2),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    layer_offsets: Optional[Tuple[float, ...]] = None,
    vias: Optional[Tuple[Optional[ComponentSpec], ...]] = (via1, via2),
    via_min_size: Tuple[Float2, ...] = ((0.2, 0.2), (0.2, 0.2)),
    via_min_gap: Tuple[Float2, ...] = ((0.1, 0.1), (0.1, 0.1)),
    via_min_enclosure: Float2 = (0.15, 0.25),
    layer_port: LayerSpec = None,
) -> Component:
    """Rectangular via array stack, with optimized dimension for vias.

    Uses inclusion, minimum width, and minimum spacing rules to place the maximum number of individual vias, each with maximum via area.

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
            ref = c << compass(size=size, layer=layer, port_type="placement")

    vias = vias or []
    c.info["vias"] = []
    for current_via, min_size, min_gap, min_enclosure in zip(
        vias, via_min_size, via_min_gap, via_min_enclosure
    ):
        if current_via is not None:
            # Optimize via
            via_type = gf.get_component(
                optimized_via(current_via, size, min_size, min_gap, min_enclosure)
            )
            c.info["vias"].append(via_type.info)

            w, h = via_type.info["size"]
            g = via_type.info["enclosure"]
            pitch_x, pitch_y = via_type.info["spacing"]

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1
            ref = c.add_array(
                via_type, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
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
        base_via: to modify
        size: of the target enclosing medium
        min_via_size: minimum size the vias can take
        min_via_gap: minimum distance between vias
        min_via_enclosure: minimum distance between edge of enclosing medium and nearest via edge
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

    return gf.partial(
        base_via,
        size=via_size,
        gap=min_via_gap,
        spacing=None,
        enclosure=min_via_enclosure,
    )


def test_via_stack_from_rules():
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


via_stack_m1_m3 = gf.partial(
    via_stack,
    layers=("M1", "M2", "M3"),
    vias=(via1, via2),
)

via_stack_slab_m3 = gf.partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "M3"),
    vias=(viac, via1, via2),
)
via_stack_npp_m1 = gf.partial(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_slab_npp_m3 = gf.partial(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_heater_mtop = via_stack_heater_m3 = gf.partial(
    via_stack, layers=("HEATER", "M2", "M3"), vias=(via1, via2)
)


if __name__ == "__main__":
    # c = via_stack_m1_m3()
    # print(c.to_dict())
    # c.show(show_ports=True)

    # c = via_stack_from_rules()
    # c = via_stack_heater_mtop()
    # c.show(show_ports=True)

    c = circular_via_stack(
        radius=20.0,
        angular_extent=45.0,
        center_angle=215.0,
        width=5.0,
        layers=("M1", "M2", "M3"),
        vias=(via1, via2),
        layer_port=None,
    )
    c.show()

    # test_via_stack_from_rules()
