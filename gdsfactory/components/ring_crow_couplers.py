from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.coupler_full import coupler_full
from gdsfactory.cross_section import strip
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_crow_couplers(
    radius: list[float] = [10.0] * 3,
    bends: list[ComponentSpec] = [bend_circular] * 3,
    ring_cross_sections: list[CrossSectionSpec] = [strip] * 3,
    couplers: list[ComponentSpec] = [coupler_full] * 4,
) -> Component:
    """Coupled ring resonators with coupler components between gaps.

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler: ring coupler spec.
        straight: straight spec.
        bend: bend spec.
        cross_section: cross_section spec.
        couplers: coupling component between rings and bus.

    .. code::

         --==ct==-- gap[N-1]   <------- couplers[N-1]
          |      |
          sl     sr ring[N-1]
          |      |
         --==cb==-- gap[N-2]   <------- couplers[N-2]

             .
             .
             .

         --==ct==--
          |      |
          sl     sr lengths_y[1], ring[1]
          |      |
         --==cb==-- gap[1]
                                <------- couplers[1]
         --==ct==--
          |      |
          sl     sr lengths_y[0], ring[0]
          |      |
         --==cb==-- gap[0]      <------- couplers[0]

          length_x
    """
    c = Component()

    couplers_refs = []
    for coupler in couplers:
        coupler_ref = (
            c.add_ref(coupler)
            if type(coupler) == gf.Component
            else c.add_ref(coupler())
        )
        couplers_refs.append(coupler_ref)

    # Input bus
    c.add_port(name="o1", port=couplers_refs[0].ports["o1"])
    c.add_port(name="o2", port=couplers_refs[0].ports["o4"])

    # Cascade rings
    for index, (r, bend, cross_section) in enumerate(
        zip(radius, bends, ring_cross_sections)
    ):
        # Add ring
        bend_c = gf.get_component(bend, radius=r, cross_section=cross_section)
        bend1 = c.add_ref(bend_c, alias=f"bot_right_bend_ring_{index}")
        bend2 = c.add_ref(bend_c, alias=f"top_right_bend_ring_{index}")
        bend3 = c.add_ref(bend_c, alias=f"top_left_bend_ring_{index}")
        bend4 = c.add_ref(bend_c, alias=f"bot_left_bend_ring_{index}")

        # We need to account for the chance that the top and bottom couplers
        # have a different length --> In this case we need to add straights
        coup1_extent = couplers_refs[index].xmax - couplers_refs[index].xmin
        coup2_extent = couplers_refs[index + 1].xmax - couplers_refs[index + 1].xmin

        if coup1_extent == coup2_extent:
            # Length of the couplers is the same -- we are good
            bend1.connect("o1", couplers_refs[index].ports["o3"])
            bend2.connect("o1", bend1.ports["o2"])
            couplers_refs[index + 1].connect("o4", bend2.ports["o2"])
            bend3.connect("o1", couplers_refs[index + 1].ports["o1"])
            bend4.connect("o1", bend3.ports["o2"])

        else:
            str_len = np.abs(coup1_extent - coup2_extent) / 2
            str_sec = gf.components.straight(
                cross_section=cross_section, length=str_len
            )

            str1 = c << str_sec
            str2 = c << str_sec

            if coup1_extent > coup2_extent:
                # The straight are connected to coupler 2
                bend1.connect("o1", couplers_refs[index].ports["o3"])
                bend2.connect("o1", bend1.ports["o2"])
                str1.connect("o1", bend2.ports["o2"])
                couplers_refs[index + 1].connect("o4", str1.ports["o2"])
                str2.connect("o1", couplers_refs[index + 1].ports["o1"])
                bend3.connect("o1", str2.ports["o2"])
                bend4.connect("o1", bend3.ports["o2"])
            else:
                # The straights are connected to coupler 1
                str1.connect("o1", couplers_refs[index].ports["o3"])
                str2.connect("o2", couplers_refs[index].ports["o2"])
                bend1.connect("o1", str1.ports["o2"])
                bend2.connect("o1", bend1.ports["o2"])
                couplers_refs[index + 1].connect("o4", bend2.ports["o2"])
                bend3.connect("o1", couplers_refs[index + 1].ports["o1"])
                bend4.connect("o1", bend3.ports["o2"])

    # Output bus
    c.add_port(name="o3", port=couplers_refs[-1].ports["o2"])
    c.add_port(name="o4", port=couplers_refs[-1].ports["o3"])

    return c


if __name__ == "__main__":
    c = ring_crow_couplers(
        couplers=[gf.components.coupler_full(coupling_length=0.01, dw=0)] * 4
    )

    c.show(show_ports=True, show_subports=False)

    print(c.named_references["bot_right_bend_ring_0"].ports)
