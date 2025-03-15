from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.components.vias.via import via
from gdsfactory.components.vias.via_stack import via_stack
from gdsfactory.cross_section import Section, rib
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionFactory,
    CrossSectionSpec,
    LayerSpec,
)

cross_section_rib = partial(
    gf.cross_section.strip,
    sections=(Section(width=2 * 2.425, layer="SLAB90", name="slab"),),
)
cross_section_pn = partial(
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
    layers=("M1", "M2", "M3"),
    vias=(
        partial(via, layer="VIAC", size=(0.1, 0.1), enclosure=0.1, pitch=0.2),
        partial(
            via,
            layer="VIA1",
            size=(0.1, 0.1),
            enclosure=0.1,
            pitch=0.2,
        ),
        None,
    ),
)


@gf.cell
def ring_double_pn(
    add_gap: float = 0.3,
    drop_gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 85,
    cross_section: CrossSectionFactory = rib,
    pn_cross_section: CrossSectionFactory = cross_section_pn,
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 2.175,
    heater_vias: ComponentSpec = _heater_vias,
    with_drop: bool = True,
    **kwargs: Any,
) -> gf.Component:
    """Returns add-drop pn ring with optional doped heater.

    Args:
        add_gap: gap to add waveguide. Bottom gap.
        drop_gap: gap to drop waveguide. Top gap.
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
        with_drop: boolean for if we include drop waveguide or not.
        kwargs: cross_section settings.

    """
    add_gap = gf.snap.snap_to_grid(add_gap, grid_factor=2)
    drop_gap = gf.snap.snap_to_grid(drop_gap, grid_factor=2)
    c = gf.Component()

    pn_cross_section_ = gf.get_cross_section(pn_cross_section, **kwargs)
    cross_section_ = gf.get_cross_section(cross_section, **kwargs)
    cross_section_ = cross_section_.copy(**kwargs)

    heater_vias = gf.get_component(heater_vias)
    undoping_angle = 180 - doping_angle

    th_waveguide_path = gf.Path()
    th_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    th_waveguide = c << th_waveguide_path.extrude(cross_section=cross_section_)
    th_waveguide.dx = 0
    th_waveguide.dy = (
        -radius
        - add_gap
        - th_waveguide.ports["o1"].width / 2
        - pn_cross_section_.width / 2
    )

    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=radius, angle=-doping_angle))
    undoped_path = gf.Path()
    undoped_path.append(gf.path.arc(radius=radius, angle=undoping_angle))

    r = gf.ComponentAllAngle()
    left_doped_ring_ref = r.create_vinst(
        doped_path.extrude(cross_section=pn_cross_section_, all_angle=True)
    )
    right_doped_ring_ref = r.create_vinst(
        doped_path.extrude(cross_section=pn_cross_section_, all_angle=True)
    )
    bottom_undoped_ring_ref = r.create_vinst(
        undoped_path.extrude(cross_section=cross_section_, all_angle=True)
    )
    top_undoped_ring_ref = r.create_vinst(
        undoped_path.extrude(cross_section=cross_section_, all_angle=True)
    )

    bottom_undoped_ring_ref.drotate(-undoping_angle / 2)
    bottom_undoped_ring_ref.dx = th_waveguide.dx

    left_doped_ring_ref.connect("o1", bottom_undoped_ring_ref.ports["o1"])
    right_doped_ring_ref.connect("o2", bottom_undoped_ring_ref.ports["o2"])
    top_undoped_ring_ref.connect("o2", left_doped_ring_ref.ports["o2"])

    ring = c.create_vinst(r)
    ring.center = (0, 0)

    drop_waveguide_dy = (
        radius
        + drop_gap
        + th_waveguide.ports["o1"].width / 2
        + pn_cross_section_.width / 2
    )

    if doped_heater:
        heater_radius = radius - doped_heater_waveguide_offset
        heater_path = gf.Path()
        heater_path.append(
            gf.path.arc(
                radius=heater_radius, angle=undoping_angle - doped_heater_angle_buffer
            )
        )

        heater = heater_path.extrude(width=0.5, layer=doped_heater_layer)

        bottom_heater_ref = c << heater
        bottom_heater_ref.drotate(-(undoping_angle - doped_heater_angle_buffer) / 2)
        bottom_heater_ref.dx = th_waveguide.dx
        bottom_heater_ref.dy = th_waveguide.dy + (
            doped_heater_waveguide_offset + doped_heater_width / 2 + add_gap
        )

        bottom_l_heater_via = c << heater_vias
        bottom_r_heater_via = c << heater_vias
        bottom_l_heater_via.dx = bottom_heater_ref.ports["o1"].dx
        bottom_l_heater_via.dy = bottom_heater_ref.ports["o1"].dy
        bottom_r_heater_via.dx = bottom_heater_ref.ports["o2"].dx
        bottom_r_heater_via.dy = bottom_heater_ref.ports["o2"].dy

        top_heater_ref = c << heater
        top_heater_ref.drotate(180 - (undoping_angle - doped_heater_angle_buffer) / 2)
        top_heater_ref.dx = th_waveguide.dx
        top_heater_ref.dy = drop_waveguide_dy - (
            doped_heater_waveguide_offset + doped_heater_width / 2 + drop_gap
        )

        top_l_heater_via = c << heater_vias
        top_r_heater_via = c << heater_vias
        top_l_heater_via.dx = top_heater_ref.ports["o1"].dx
        top_l_heater_via.dy = top_heater_ref.ports["o1"].dy
        top_r_heater_via.dx = top_heater_ref.ports["o2"].dx
        top_r_heater_via.dy = top_heater_ref.ports["o2"].dy

    c.add_port("o1", port=th_waveguide.ports["o1"])
    c.add_port("o2", port=th_waveguide.ports["o2"])

    c.add_port(name="htr_top_sig", port=top_l_heater_via["e2"])
    c.add_port(name="htr_top_gnd", port=top_r_heater_via["e2"])
    c.add_port(name="htr_bot_sig", port=bottom_l_heater_via["e2"])
    c.add_port(name="htr_bot_gnd", port=bottom_r_heater_via["e2"])

    if with_drop:
        drop_waveguide_path = gf.Path()
        drop_waveguide_path.append(
            gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
        )
        drop_waveguide = c << drop_waveguide_path.extrude(cross_section=cross_section_)
        drop_waveguide.dx = 0
        drop_waveguide.dy = drop_waveguide_dy
        c.add_port("o3", port=drop_waveguide.ports["o2"])
        c.add_port("o4", port=drop_waveguide.ports["o1"])
    c.flatten()
    return c


@gf.cell
def ring_single_pn(
    gap: float = 0.3,
    radius: float = 5.0,
    doping_angle: float = 250,
    cross_section: CrossSectionSpec = rib,
    pn_cross_section: CrossSectionSpec = cross_section_pn,
    doped_heater: bool = True,
    doped_heater_angle_buffer: float = 10,
    doped_heater_layer: LayerSpec = "NPP",
    doped_heater_width: float = 0.5,
    doped_heater_waveguide_offset: float = 1.175,
    heater_vias: ComponentSpec = _heater_vias,
    pn_vias: ComponentSpec = "via_stack_slab_m3",
    pn_vias_width: float = 3,
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
        pn_vias: components specifications for pn vias.
        pn_vias_width: width of pn vias.
    """
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)
    c = gf.Component()

    undoping_angle = 360 - doping_angle

    pn_xs = gf.get_cross_section(pn_cross_section)
    bus_waveguide_path = gf.Path()
    bus_waveguide_path.append(
        gf.path.straight(length=2 * radius * np.sin(np.pi / 360 * undoping_angle))
    )
    bus_waveguide = c << bus_waveguide_path.extrude(cross_section=cross_section)
    bus_waveguide.dx = 0
    bus_waveguide.dy = (
        -radius
        - gap
        - bus_waveguide.ports["o1"].width / 2
        - pn_xs.width / 2
        + 0.576  # adjust gap # TODO: remove this
    )

    r = gf.Component()
    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=radius, angle=-doping_angle))
    undoped_path = gf.Path()
    undoped_path.append(gf.path.arc(radius=radius, angle=undoping_angle))

    doped_ring_ref = r << doped_path.extrude(cross_section=pn_xs, all_angle=False)
    undoped_ring_ref = r << undoped_path.extrude(
        cross_section=cross_section, all_angle=False
    )
    undoped_ring_ref.drotate(-undoping_angle / 2)
    undoped_ring_ref.center = (0, 0)
    doped_ring_ref.connect("o1", undoped_ring_ref.ports["o1"])

    via = gf.get_component(pn_vias, size=(pn_vias_width, pn_vias_width))
    gnd = r << via
    gnd.dx = doped_ring_ref.ports["e1_top"].dx
    gnd.dy = doped_ring_ref.ports["e1_top"].dy

    sig = r << via
    sig.dx = doped_ring_ref.ports["e2_bot"].dx
    sig.dy = doped_ring_ref.ports["e2_bot"].dy
    r.add_port("sig", port=sig["e2"])
    r.add_port("gnd", port=gnd["e2"])

    ring = c << r
    ring.center = (0, 0)

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
            + radius / 4
        )

        heater_vias = gf.get_component(heater_vias)

        bottom_l_heater_via = c << heater_vias
        bottom_r_heater_via = c << heater_vias
        bottom_l_heater_via.dxmin = bottom_heater_ref.ports["o1"].dx
        bottom_l_heater_via.dymax = bottom_heater_ref.ports["o1"].dy

        bottom_r_heater_via.dxmax = bottom_heater_ref.ports["o2"].dx
        bottom_r_heater_via.dymax = bottom_heater_ref.ports["o2"].dy

        c.add_port(name="heater_sig", port=bottom_l_heater_via["e4"])
        c.add_port(name="heater_gnd", port=bottom_r_heater_via["e4"])

    c.add_port("o1", port=bus_waveguide.ports["o1"])
    c.add_port("o2", port=bus_waveguide.ports["o2"])
    c.add_ports(ring.ports)
    c.flatten()
    return c


if __name__ == "__main__":
    c = ring_double_pn(radius=5)
    c.pprint_ports()
    c.show()
