from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.components.via import via
from gdsfactory.components.via_stack import via_stack
from gdsfactory.cross_section import Section
from gdsfactory.typings import ComponentSpec, CrossSection, LayerSpec


@gf.cell
def ring_double_pn(
    add_gap: float = 0.3,
    drop_gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 85,
    cross_section: CrossSection = partial(
        gf.cross_section.strip,
        sections=(Section(width=2 * 2.425, layer="SLAB90", name="slab"),),
    ),
    pn_cross_section: CrossSection = partial(
        gf.cross_section.pn,
        width_doping=2.425,
        width_slab=2 * 2.425,
        layer_via="VIAC",
        width_via=0.5,
        layer_metal="M1",
        width_metal=0.5,
    ),
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 2.175,
    heater_vias: ComponentSpec = partial(
        via_stack,
        size=(0.5, 0.5),
        layers=("M1", "M2"),
        vias=(
            partial(
                via,
                layer="VIAC",
                size=(0.1, 0.1),
                spacing=(0.2, 0.2),
                enclosure=0.1,
            ),
            partial(
                via,
                layer="VIA1",
                size=(0.1, 0.1),
                spacing=(0.2, 0.2),
                enclosure=0.1,
            ),
        ),
    ),
    **kwargs,
) -> gf.Component:
    """Returns add-drop pn ring with optional doped heater.

    Args:
        add_gap: gap to add waveguide.
        drop_gap: gap to drop waveguide.
        radius: for the bend and coupler.
        doping_angle: angle in degrees representing portion of ring that is doped.
        length_x: ring coupler length.
        length_y: vertical straight length.
        cross_section: cross_section spec for non-PN doped rib waveguide sections.
        pn_cross_section: cross section of pn junction.
        doped_heater: boolean for if we include doped heater or not.
        doped_heater_angle_buffer: angle in degrees buffering heater from pn junction.
        doped_heater_layer: doping layer for heater.
        doped_heater_width: width of doped heater.
        doped_heater_waveguide_offset: distance from the center of the ring waveguide to the center of the doped heater.
        heater_vias: components specifications for heater vias
        kwargs: cross_section settings.
    """

    add_gap = gf.snap.snap_to_grid(add_gap, nm=2)
    drop_gap = gf.snap.snap_to_grid(drop_gap, nm=2)
    c = gf.Component()

    undoping_angle = 180 - doping_angle

    add_waveguide_path = gf.Path()
    add_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    add_waveguide = c << add_waveguide_path.extrude(cross_section=cross_section)
    add_waveguide.x = 0
    add_waveguide.y = 0

    drop_waveguide_path = gf.Path()
    drop_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    drop_waveguide = c << drop_waveguide_path.extrude(cross_section=cross_section)
    drop_waveguide.x = 0

    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=radius, angle=-doping_angle))
    undoped_path = gf.Path()
    undoped_path.append(gf.path.arc(radius=radius, angle=undoping_angle))
    left_doped_ring_ref = c << doped_path.extrude(cross_section=pn_cross_section)
    right_doped_ring_ref = c << doped_path.extrude(cross_section=pn_cross_section)
    bottom_undoped_ring_ref = c << undoped_path.extrude(cross_section=cross_section)
    top_undoped_ring_ref = c << undoped_path.extrude(cross_section=cross_section)

    bottom_undoped_ring_ref.rotate(-undoping_angle / 2)
    bottom_undoped_ring_ref.ymin = (
        add_waveguide.ymin + add_waveguide.ports["o1"].width + add_gap
    )
    bottom_undoped_ring_ref.x = add_waveguide.x

    left_doped_ring_ref.connect("o1", bottom_undoped_ring_ref.ports["o1"])
    right_doped_ring_ref.connect("o2", bottom_undoped_ring_ref.ports["o2"])
    top_undoped_ring_ref.connect("o2", left_doped_ring_ref.ports["o2"])

    drop_waveguide.y = (
        2 * radius
        + add_gap
        + drop_gap
        + add_waveguide.ports["o1"].width / 2
        + top_undoped_ring_ref.ports["o1"].width
        + drop_waveguide.ports["o1"].width / 2
    )

    if doped_heater:
        heater_radius = radius - doped_heater_waveguide_offset
        heater_path = gf.Path()
        heater_path.append(
            gf.path.arc(
                radius=heater_radius, angle=undoping_angle - doped_heater_angle_buffer
            )
        )

        top_heater_ref = c << heater_path.extrude(width=0.5, layer=doped_heater_layer)
        top_heater_ref.rotate(180 - (undoping_angle - doped_heater_angle_buffer) / 2)
        top_heater_ref.x = add_waveguide.x
        top_heater_ref.ymax = drop_waveguide.y - (
            doped_heater_waveguide_offset + doped_heater_width / 2 + drop_gap
        )

        top_left_heater_via = c << heater_vias()
        top_left_heater_via.rotate(top_heater_ref.ports["o2"].orientation)

        deltax = -abs(top_heater_ref.ports["o2"].x - top_left_heater_via.ports["e3"].x)
        deltay = abs(top_heater_ref.ports["o2"].y - top_left_heater_via.ports["e3"].y)
        top_left_heater_via.move((deltax, deltay))

        top_right_heater_via = c << heater_vias()
        top_right_heater_via.rotate(top_heater_ref.ports["o1"].orientation)

        deltax = abs(top_heater_ref.ports["o1"].x - top_right_heater_via.ports["e3"].x)
        deltay = abs(top_heater_ref.ports["o1"].y - top_right_heater_via.ports["e3"].y)
        top_right_heater_via.move((deltax, deltay))

        bottom_heater_ref = c << heater_path.extrude(
            width=0.5, layer=doped_heater_layer
        )
        bottom_heater_ref.rotate(-(undoping_angle - doped_heater_angle_buffer) / 2)
        bottom_heater_ref.x = add_waveguide.x
        bottom_heater_ref.ymin = (
            doped_heater_waveguide_offset + doped_heater_width / 2 + add_gap
        )

        bottom_left_heater_via = c << heater_vias()
        bottom_left_heater_via.rotate(bottom_heater_ref.ports["o1"].orientation)

        deltax = -abs(
            bottom_heater_ref.ports["o1"].x - bottom_left_heater_via.ports["e3"].x
        )
        deltay = abs(
            bottom_heater_ref.ports["o1"].y - bottom_left_heater_via.ports["e3"].y
        )
        bottom_left_heater_via.move((deltax, deltay))

        bottom_right_heater_via = c << heater_vias()
        bottom_right_heater_via.rotate(bottom_heater_ref.ports["o2"].orientation)

        deltax = abs(
            bottom_heater_ref.ports["o2"].x - bottom_right_heater_via.ports["e3"].x
        )
        deltay = abs(
            bottom_heater_ref.ports["o2"].y - bottom_right_heater_via.ports["e3"].y
        )
        bottom_right_heater_via.move((deltax, deltay))

    c.add_port("o1", port=add_waveguide.ports["o1"])
    c.add_port("o2", port=add_waveguide.ports["o2"])
    c.add_port("o3", port=drop_waveguide.ports["o2"])
    c.add_port("o4", port=drop_waveguide.ports["o1"])

    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)
    # c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    # print(c.ports)

    # c = gf.routing.add_fiber_array(ring_single)
    c = ring_double_pn(width=0.5)
    c.show(show_ports=True)

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show(show_ports=True)
