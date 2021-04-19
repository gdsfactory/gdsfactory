from pp.cell import cell
from pp.components.bend_circular import bend_circular
from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.components.straight import straight
from pp.components.taper import taper
from pp.netlist_to_gds import netlist_to_component
from pp.snap import assert_on_2nm_grid


@cell
def ring_with_taper(**kwargs):
    """Ring single bus

    Args:
        bend_radius=5
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

      import pp

      c = pp.components.ring_single_bus(gap=0.2, length_x=10, length_y=5, bend_radius=5)
      c.plot()

    """
    components, connections, ports_map = ring_with_taper_netlist(**kwargs)
    component = netlist_to_component(components, connections, ports_map)
    return component


def ring_with_taper_netlist(
    bend_radius=5,
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
    bend = bend90_factory(radius=bend_radius, width=wg_width)
    cpl_bend = coupler90_factory(bend_radius=bend_radius, width=wg_width, gap=gap)
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
        ("CL", "W0", "CS", "W0"),
        ("CS", "E0", "CR", "W0"),
        ("CR", "N0", "T1", "1"),
        ("T1", "2", "VR", "W0"),
        ("VR", "E0", "T2", "2"),
        ("T2", "1", "BR", "N0"),
        ("BR", "W0", "H1", "E0"),
        ("H1", "W0", "BL", "W0"),
        ("BL", "N0", "VL", "E0"),
        ("VL", "W0", "CL", "N0"),
    ]

    ports_map = {"W0": ("CL", "E0"), "E0": ("CR", "E0")}

    return components, connections, ports_map


if __name__ == "__main__":
    # c = ring_with_taper(bend_radius=5.0, length_x=2, length_y=4, gap=0.2)
    c = ring_with_taper(
        bend_radius=5.0, length_x=2, length_y=4, gap=0.2, taper_width=1, wg_width=0.7
    )
    c.show()
