from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.components.via import via
from gdsfactory.components.via_stack import via_stack
from gdsfactory.cross_section import Section
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

_pn_cross_section = partial(
    gf.cross_section.pn,
    width_doping=2.425,
    width_slab=2 * 2.425,
    layer_via="VIAC",
    width_via=0.5,
    layer_metal="M1",
    width_metal=0.5,
)

_heater_vias = partial(
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
)

_cross_section = partial(
    gf.cross_section.strip,
    sections=(Section(width=2 * 2.425, layer="SLAB90", name="slab"),),
)


@gf.cell
def ring_single_pn(
    gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 250,
    cross_section: CrossSectionSpec = _cross_section,
    pn_cross_section: CrossSectionSpec = _pn_cross_section,
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 2.175,
    heater_vias: ComponentSpec = _heater_vias,
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
    """
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)
    c = gf.Component()

    undoping_angle = 360 - doping_angle

    pn_cross_section = gf.get_cross_section(pn_cross_section)
    bus_waveguide_path = gf.Path()
    bus_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    bus_waveguide = c << bus_waveguide_path.extrude(cross_section=cross_section)
    bus_waveguide.dx = 0
    bus_waveguide.dy = (
        -radius
        - gap
        - bus_waveguide.ports["o1"].dwidth / 2
        - pn_cross_section.width / 2
    )

    r = gf.ComponentAllAngle()
    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=radius, angle=-doping_angle))
    undoped_path = gf.Path()
    undoped_path.append(gf.path.arc(radius=radius, angle=undoping_angle))

    doped_ring_ref = r << doped_path.extrude(
        cross_section=pn_cross_section, all_angle=True
    )
    undoped_ring_ref = r << undoped_path.extrude(
        cross_section=cross_section, all_angle=True
    )
    undoped_ring_ref.drotate(-undoping_angle / 2)
    undoped_ring_ref.dcenter = (0, 0)  # type: ignore[assignment]
    doped_ring_ref.connect("o1", undoped_ring_ref.ports["o1"])
    ring = c.create_vinst(r)
    ring.dcenter = (0, 0)  # type: ignore[assignment]

    if doped_heater:
        heater_radius = radius - doped_heater_waveguide_offset
        heater_path = gf.Path()
        heater_path.append(
            gf.path.arc(
                radius=heater_radius, angle=undoping_angle - doped_heater_angle_buffer
            )
        )

        bottom_heater_ref = c << heater_path.extrude(
            width=0.5, layer=doped_heater_layer
        )
        bottom_heater_ref.drotate(-(undoping_angle - doped_heater_angle_buffer) / 2)
        bottom_heater_ref.dx = bus_waveguide.dx
        bottom_heater_ref.dy = (
            bus_waveguide.dy
            + doped_heater_waveguide_offset
            + doped_heater_width / 2
            + gap
        )

        heater_vias = gf.get_component(heater_vias)

        bottom_l_heater_via = c << heater_vias
        bottom_r_heater_via = c << heater_vias
        bottom_l_heater_via.connect(
            "e3",
            bottom_heater_ref.ports["o1"],
            allow_layer_mismatch=True,
            allow_type_mismatch=True,
        )
        bottom_r_heater_via.connect(
            "e3",
            bottom_heater_ref.ports["o2"],
            allow_layer_mismatch=True,
            allow_type_mismatch=True,
        )

    c.add_port("o1", port=bus_waveguide.ports["o1"])
    c.add_port("o2", port=bus_waveguide.ports["o2"])
    c.flatten()
    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)
    # c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    # print(c.ports)

    # c = gf.routing.add_fiber_array(ring_single)
    c = ring_single_pn()
    c.show()

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show( )
