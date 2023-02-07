from __future__ import annotations

from typing import List

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.coupler_full import coupler_full
from gdsfactory.cross_section import strip
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_crow_couplers(
    radius: List[float] = [10.0] * 3,
    bends: List[ComponentSpec] = [bend_circular] * 3,
    ring_cross_sections: List[CrossSectionSpec] = [strip] * 3,
    couplers: List[ComponentSpec] = [coupler_full] * 4,
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

        bend1.connect("o1", couplers_refs[index].ports["o3"])
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
