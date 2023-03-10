"""
Generates a ring resonator based on a series of given cross sections.

This is useful to generate interleaved junction rings or rings with relatively
complex junction profiles
"""

from functools import partial
import numpy as np
import gdsfactory as gf
from typing import Dict, List, Optional
from gdsfactory.typings import CrossSectionSpec
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight

def_dict = {"A": "rib", "B": "strip"}


@gf.cell
def ring_section_based(
    gap: float = 0.3,
    radius: float = 5.0,
    add_drop: bool = False,
    cross_sections: Dict = def_dict,
    cross_sections_sequence: str = "AB",
    cross_sections_angles: Optional[List[float]] = (6, 6),
    init_cross_section: Optional[CrossSectionSpec] = None,
    init_angle: Optional[float] = 10.0,
    init_section_at_drop: bool = True,
    bus_cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Returns a ring made of the specified cross sections.

    We start with init_cross section if indicated, then repeat the sequence in
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
            cross_sections_sequence list (deg). If not indicated, then we assume that
            the sequence is only repeated once and calculate the necessary angular
            length
        init_cross_section: it is likely that the cross section at the ring-bus junction
            is different than the sequence we want to repeat. If that's the case, then
            here you indicate the initial cross section.
        init_angle: angular extent of the initial cross section (deg)
        init_section_at_drop: if True, the initial section is placed at both ring-bus
            junctions (only applies if add_drop = True)
        bus_cross_section: cross section for the bus waveguide.
    """

    c = gf.Component()

    # First of all we need to do a bunch of checks
    angular_extent_sequence = 360

    # See if we need to add initial cross sections
    if init_cross_section is not None:
        init_xs = gf.get_cross_section(init_cross_section)
        angular_extent_sequence -= init_angle

        if add_drop and init_section_at_drop:
            angular_extent_sequence -= init_angle

    n_sections = len(cross_sections_sequence)
    if cross_sections_angles is None:
        # Calculate the extents of the sequences so they fit
        if add_drop and init_section_at_drop:
            cross_sections_angles = [
                angular_extent_sequence / (2 * n_sections)
            ] * n_sections
        else:
            cross_sections_angles = [angular_extent_sequence / n_sections] * n_sections

    # Now make sure that the specified angular extents of the sections
    # are compatible with the ring extent (360 degree)
    sing_seq_angular_extent = np.sum(cross_sections_angles)
    # Make sure we can fit an integer number of sequences into the
    # ring circumference
    if add_drop and init_section_at_drop:
        if not (angular_extent_sequence / (sing_seq_angular_extent / 2)).is_integer():
            raise ValueError(
                "The specified sequence angles do not result in an integer number of sequences fitting in the ring."
            )
        else:
            n_repeats_seq = int(angular_extent_sequence / (2 * sing_seq_angular_extent))
    else:
        if not (angular_extent_sequence / sing_seq_angular_extent).is_integer():
            raise ValueError(
                "The specified sequence angles do not result in an integer number of sequences fitting in the ring."
            )
        else:
            n_repeats_seq = int(angular_extent_sequence / sing_seq_angular_extent)

    # Now we are ready to construct the ring

    # We need to create a circular bend for each section
    sections_dict = dict()

    for i, key in enumerate(cross_sections.keys()):
        ang = cross_sections_angles[i]
        xsec = cross_sections[key]

        b = bend_circular(angle=ang, with_bbox=False, cross_section=xsec, radius=radius)

        sections_dict[key] = (b, "o1", "o2")

    if init_cross_section is not None:
        b = bend_circular(
            angle=init_angle, with_bbox=False, cross_section=init_xs, radius=radius
        )
        if "0" in sections_dict:
            raise ValueError(
                "Please do not have '0' as a key for the cross_sections dict"
            )
        sections_dict["0"] = (b, "o1", "o2")

    # Now we just need to generate a chain of characters
    # to creae the sequence

    sequence = ""

    if init_cross_section is not None:
        sequence += "0"

    sequence += cross_sections_sequence * n_repeats_seq

    if add_drop and init_section_at_drop:
        sequence = sequence * 2

    print(sequence)
    print(sections_dict)

    ring = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=sections_dict
    )

    r = c << ring

    # Rotate so that the first section is centered at the add bus
    if init_cross_section is not None:
        ring = ring.rotate(
            -init_angle / 2
        )  # also change the component for later bound extraction
        r.rotate(-init_angle / 2)
    else:
        ring = ring.rotate(-cross_sections_angles[0] / 2)
        r.rotate(-cross_sections_angles[0] / 2)

    # Add bus waveguides

    # Figure out main waveguiding layer of the ring at the ring-bus interface
    input_xs_layer = (
        gf.get_cross_section(init_cross_section).layer
        if init_cross_section
        else gf.get_cross_section(cross_sections[cross_sections_sequence[0]]).layer
    )
    ring_guide = ring.extract([input_xs_layer])

    print(ring_guide.ymin)

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

    return c


if __name__ == "__main__":
    from gdsfactory.cross_section import rib

    c = gf.Component()

    # rib = c << straight(length = 100, cross_section="rib")
    # b_rib = c << bend_circular(angle = 10,
    #                       with_bbox = True,
    #                       cross_section = "slot",
    #                       radius=5)
    # strip = c << straight(length = 100, cross_section="strip")

    # strip.ymax = b_rib.ymin - 10

    c << ring_section_based(
        gap=0.3,
        radius=5.0,
        add_drop=True,
        cross_sections={"A": "rib", "B": "slot"},
        cross_sections_sequence="AB",
        cross_sections_angles=[17, 17],
        init_cross_section=partial(rib, width=0.65),
        init_angle=10.0,
        init_section_at_drop=True,
        bus_cross_section="strip",
    )

    """
    b = c << bend_circular(angle = 15,
                        with_bbox = True,
                        cross_section = "strip",)
    print(b.ports)
    b.rotate(-7.5)
    """

    c.show()
