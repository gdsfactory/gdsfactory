from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.components.via import via
from gdsfactory.components.via_stack import via_stack
from gdsfactory.cross_section import Section
from gdsfactory.typings import ComponentSpec, CrossSection, LayerSpec


@gf.cell
def ring_single_pn(
    gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 250,
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
    """Returns single pn ring with optional doped heater.

    Args:
        gap: gap between for coupler.
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
        heater_vias: components specifications for heater vias.
        kwargs: cross_section settings.
    """

    gap = gf.snap.snap_to_grid(gap, nm=2)
    c = gf.Component()

    undoping_angle = 360 - doping_angle

    bus_waveguide_path = gf.Path()
    bus_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    bus_waveguide = c << bus_waveguide_path.extrude(cross_section=cross_section)
    bus_waveguide.x = 0

    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=radius, angle=-doping_angle))
    undoped_path = gf.Path()
    undoped_path.append(gf.path.arc(radius=radius, angle=undoping_angle))
    doped_ring_ref = c << doped_path.extrude(cross_section=pn_cross_section)
    undoped_ring_ref = c << undoped_path.extrude(cross_section=cross_section)

    undoped_ring_ref.rotate(-undoping_angle / 2)
    undoped_ring_ref.ymin = bus_waveguide.ymin + bus_waveguide.ports["o1"].width + gap
    undoped_ring_ref.x = bus_waveguide.x

    doped_ring_ref.connect("o1", undoped_ring_ref.ports["o1"])

    if doped_heater:
        heater_radius = radius - doped_heater_waveguide_offset
        heater_path = gf.Path()
        heater_path.append(
            gf.path.arc(
                radius=heater_radius, angle=undoping_angle - doped_heater_angle_buffer
            )
        )

        heater_ref = c << heater_path.extrude(width=0.5, layer=doped_heater_layer)
        heater_ref.rotate(-(undoping_angle - doped_heater_angle_buffer) / 2)
        heater_ref.x = bus_waveguide.x
        heater_ref.ymin = doped_heater_waveguide_offset + doped_heater_width / 2 + gap

        left_heater_via = c << heater_vias()
        left_heater_via.rotate(heater_ref.ports["o1"].orientation)

        deltax = -abs(heater_ref.ports["o1"].x - left_heater_via.ports["e3"].x)
        deltay = abs(heater_ref.ports["o1"].y - left_heater_via.ports["e3"].y)
        left_heater_via.move((deltax, deltay))

        right_heater_via = c << heater_vias()
        right_heater_via.rotate(heater_ref.ports["o2"].orientation)

        deltax = abs(heater_ref.ports["o2"].x - right_heater_via.ports["e3"].x)
        deltay = abs(heater_ref.ports["o2"].y - right_heater_via.ports["e3"].y)
        right_heater_via.move((deltax, deltay))

    c.add_port("o1", port=bus_waveguide.ports["o1"])
    c.add_port("o2", port=bus_waveguide.ports["o2"])

    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)
    # c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    # print(c.ports)

    # c = gf.routing.add_fiber_array(ring_single)
    c = ring_single_pn()
    print([i.name for i in c.get_dependencies()])
    c.show(show_ports=True)

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show(show_ports=True)
