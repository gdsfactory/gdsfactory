import gdsfactory as gf
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.coupler90 import coupler90
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper
from gdsfactory.snap import assert_on_2nm_grid


@gf.cell
def ring_with_taper(**kwargs):
    """Ring single bus

    Args:
        radius=5
        length_x=1
        length_y=1
        gap=0.2
        taper_factory=taper
        bend90_factory=bend_circular
        coupler90_factory=coupler90
        straight_factory=straight
        cpl_straight_factory=coupler_straight
        wg_width=0.5


    .. code::

         --length_x-
         |         |
         |       length_y
         |         |
       ---===gap===---

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.ring_single_bus(gap=0.2, length_x=10, length_y=5, radius=5)
      c.plot()

    """
    components, connections, ports_map = ring_with_taper_netlist(**kwargs)
    component = gf.component_from.netlist(components, connections, ports_map)
    return component


def ring_with_taper_netlist(
    radius=5,
    length_x=1,
    length_y=0,
    gap=0.2,
    taper_width=1.0,
    taper_length=10,
    taper_factory=taper,
    bend90_factory=bend_circular,
    coupler90_factory=coupler90,
    straight_factory=straight,
    cpl_straight_factory=coupler_straight,
    wg_width=0.5,
):
    """
    .. code::

         BL--H1--BR
         |       |
         |       T2
         |       VR length_y
         VL      T1
         |       |
        -CL==CS==CR-
    """
    assert_on_2nm_grid(gap)

    taper = taper_factory(length=taper_length, width1=wg_width, width2=taper_width)
    bend = bend90_factory(radius=radius, width=wg_width)
    cpl_bend = coupler90_factory(radius=radius, width=wg_width, gap=gap)
    cpl_straight = cpl_straight_factory(length=length_x, gap=gap, width=wg_width)
    h1 = straight_factory(length=length_x, width=wg_width)
    vl = straight_factory(length=length_y + 2 * taper_length, width=wg_width)
    vr = straight_factory(length=length_y, width=taper_width)

    components = {
        "CL": (cpl_bend, "mirror_y"),
        "CR": (cpl_bend, "None"),
        "BR": (bend, "mirror_x"),
        "BL": (bend, "R180"),
        "CS": (cpl_straight, "None"),
        "H1": (h1, "None"),
        "T1": (taper, "R90"),
        "T2": (taper, "R270"),
        "VL": (vl, "R90"),
        "VR": (vr, "R90"),
    }
    connections = [
        ("CL", 1, "CS", 1),
        ("CS", 2, "CR", 1),
        ("CR", 2, "T1", 1),
        ("T1", "2", "VR", 1),
        ("VR", 2, "T2", 2),
        ("T2", "1", "BR", 2),
        ("BR", 1, "H1", 2),
        ("H1", 1, "BL", 1),
        ("BL", 2, "VL", 2),
        ("VL", 1, "CL", 2),
    ]

    ports_map = {1: ("CL", 2), 2: ("CR", 2)}

    return components, connections, ports_map


if __name__ == "__main__":
    # c = ring_with_taper(radius=5.0, length_x=2, length_y=4, gap=0.2)
    c = ring_with_taper(
        radius=5.0, length_x=2, length_y=4, gap=0.2, taper_width=1, wg_width=0.7
    )
    c.show()
