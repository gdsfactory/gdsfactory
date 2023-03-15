"""
Generates a ring resonator based on a series of given cross sections.

This is useful to generate interleaved junction rings or rings with relatively
complex junction profiles
"""

from functools import partial
from typing import Dict, Optional

import numpy as np
import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight

def_dict = {"A": "rib", "B": "strip"}
def_ang_dict = {"A": 6, "B": 6}


@gf.cell
def ring_section_based(
    gap: float = 0.3,
    radius: float = 5.0,
    add_drop: bool = False,
    cross_sections: Dict = def_dict,
    cross_sections_sequence: str = "AB",
    cross_sections_angles: Optional[Dict] = def_ang_dict,
    start_cross_section: Optional[CrossSectionSpec] = None,
    start_angle: Optional[float] = 10.0,
    start_section_at_drop: bool = True,
    bus_cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Returns a ring made of the specified cross sections.

    We start with start_cross section if indicated, then repeat the sequence in
    cross_section_sequence until the whole ring is filled.

    Args:
        gap: bus waveguide - ring gap.
        radius: ring radius.
        add_drop: if True, we draw an add-drop ring
        cross_sections: dictionary of cross sections to add consecutively
            to the ring until the ring is filled. Keys should be single character.
        cross_sections_sequence: sequence to follow filling the ring.
            Ex: "AB" means we will put first section A, then section B,
            then section A again... until the ring is filled
        cross_sections_angles: angular extent of each cross section in the
            cross_sections dictionary (deg). If not indicated, then we assume that
            the sequence is only repeated once and calculate the necessary angular
            length
        start_cross_section: it is likely that the cross section at the ring-bus junction
            is different than the sequence we want to repeat. If that's the case, then
            here you indicate the initial cross section.
        start_angle: angular extent of the initial cross section (deg)
        start_section_at_drop: if True, the initial section is placed at both ring-bus
            junctions (only applies if add_drop = True)
        bus_cross_section: cross section for the bus waveguide.
    """

    c = gf.Component()

    # First of all we need to do a bunch of checks
    angular_extent_sequence = 360

    # See if we need to add initial cross sections
    if start_cross_section is not None:
        start_xs = gf.get_cross_section(start_cross_section)
        angular_extent_sequence -= start_angle

        if add_drop and start_section_at_drop:
            angular_extent_sequence -= start_angle

    if cross_sections_angles is None:
        n_sections = len(cross_sections_sequence)

        if add_drop and start_section_at_drop:
            sing_sec_ang = angular_extent_sequence / (2 * n_sections)
        else:
            sing_sec_ang = angular_extent_sequence / n_sections

        cross_sections_angles = {elem: sing_sec_ang for elem in cross_sections_sequence}
    # Now make sure that the specified angular extents of the sections
    # are compatible with the ring extent (360 degree)
    sing_seq_angular_extent = np.sum(
        [cross_sections_angles[sec] for sec in cross_sections_sequence]
    )
    # Make sure we can fit an integer number of sequences into the
    # ring circumference
    if add_drop and start_section_at_drop:
        if (angular_extent_sequence / (sing_seq_angular_extent / 2)).is_integer():
            n_repeats_seq = int(angular_extent_sequence / (2 * sing_seq_angular_extent))
        else:
            raise ValueError(
                "The specified sequence angles do not result in an integer "
                "number of sequences fitting in the ring."
            )
    elif not (angular_extent_sequence / sing_seq_angular_extent).is_integer():
        raise ValueError(
            "The specified sequence angles do not result in an integer number "
            "of sequences fitting in the ring."
        )
    else:
        n_repeats_seq = int(angular_extent_sequence / sing_seq_angular_extent)

    # Now we are ready to construct the ring

    # We need to create a circular bend for each section
    sections_dict = {}

    for key in cross_sections.keys():
        ang = cross_sections_angles[key]
        xsec = cross_sections[key]

        b = bend_circular(angle=ang, with_bbox=False, cross_section=xsec, radius=radius)

        sections_dict[key] = (b, "o1", "o2")

    if start_cross_section is not None:
        b = bend_circular(
            angle=start_angle, with_bbox=False, cross_section=start_xs, radius=radius
        )
        if "0" in sections_dict:
            raise ValueError(
                "Please do not have '0' as a key for the cross_sections dict"
            )
        sections_dict["0"] = (b, "o1", "o2")

    # Now we just need to generate a chain of characters
    # to creae the sequence

    sequence = ""

    if start_cross_section is not None:
        sequence += "0"

    sequence += cross_sections_sequence * n_repeats_seq

    if add_drop and start_section_at_drop:
        sequence *= 2

    ring = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=sections_dict
    )

    r = c << ring

    # Rotate so that the first section is centered at the add bus
    if start_cross_section is not None:
        ring = ring.rotate(
            -start_angle / 2
        )  # also change the component for later bound extraction
        r.rotate(-start_angle / 2)
        ring_center = [
            -radius * np.sin(np.radians(-start_angle / 2)),
            radius * np.cos(np.radians(-start_angle / 2)),
        ]
    else:
        ring = ring.rotate(-cross_sections_angles[sequence[0]] / 2)
        r.rotate(-cross_sections_angles[sequence[0]] / 2)
        ring_center = [
            -radius * np.sin(np.radians(-cross_sections_angles[sequence[0]] / 2)),
            radius * np.cos(np.radians(-cross_sections_angles[sequence[0]] / 2)),
        ]
    c.info["ring_center"] = gf.snap.snap_to_grid(ring_center)
    # Add bus waveguides

    # Figure out main waveguiding layer of the ring at the ring-bus interface
    input_xs_layer = (
        gf.get_cross_section(start_cross_section).layer
        if start_cross_section
        else gf.get_cross_section(cross_sections[cross_sections_sequence[0]]).layer
    )
    ring_guide = ring.extract([input_xs_layer])

    # Add bus waveguides
    s = straight(length=ring.xsize, cross_section=bus_cross_section)

    # Figure out main waveguiding layer of the bus at the ring-bus interface
    input_xs_width = gf.get_cross_section(bus_cross_section).width

    s_add = c << s
    s_add.x = r.x
    s_add.ymax = ring_guide.ymin - gap + s.ysize / 2 - input_xs_width / 2

    if add_drop:
        s_drop = c << s
        s_drop.x = r.x
        s_drop.ymin = ring_guide.ymax + gap - s.ysize / 2 + input_xs_width / 2

    # Add ports
    c.add_port("o1", port=s_add.ports["o1"])
    c.add_port("o2", port=s_add.ports["o2"])
    if add_drop:
        c.add_port("o3", port=s_drop.ports["o1"])
        c.add_port("o4", port=s_drop.ports["o2"])

    return c


if __name__ == "__main__":
    from gdsfactory.cross_section import rib

    # rib = c << straight(length = 100, cross_section="rib")
    # b_rib = c << bend_circular(angle = 10,
    #                       with_bbox = True,
    #                       cross_section = "slot",
    #                       radius=5)
    # strip = c << straight(length = 100, cross_section="strip")
    # strip.ymax = b_rib.ymin - 10

    c = ring_section_based(
        gap=0.3,
        radius=5.0,
        add_drop=True,
        cross_sections={"A": "rib", "B": "slot"},
        cross_sections_sequence="AB",
        cross_sections_angles={"A": 17, "B": 17},
        start_cross_section=partial(rib, width=0.65),
        start_angle=10.0,
        start_section_at_drop=True,
        bus_cross_section="strip",
    )

    # b = c << bend_circular(angle = 15,
    #                     with_bbox = True,
    #                     cross_section = "strip",)
    # print(b.ports)
    # b.rotate(-7.5)
    c = ring_section_based(add_drop=True)

    c.show(show_ports=True)
    print(c.info["ring_center"])
