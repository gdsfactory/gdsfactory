import pp
from pp.components.bend_circular import bend_circular
from pp.components.coupler90 import coupler90
from pp.components.waveguide import waveguide
from pp.components.coupler_straight import coupler_straight
from pp.netlist_to_gds import netlist_to_component
from pp.name import autoname
from pp.drc import assert_on_2nm_grid


@autoname
def ring_single_bus(**kwargs):
    """ Ring single bus

    Args:
        bend_radius=5
        length_x=1
        length_y=1
        gap=0.2
        bend90_factory=bend_circular
        coupler90_factory=coupler90
        straight_factory=waveguide
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

      c = pp.c.ring_single_bus(gap=0.2, length_x=10, length_y=5, bend_radius=5)
      pp.plotgds(c)

    """
    components, connections, ports_map = ring_single_bus_netlist(**kwargs)
    component = netlist_to_component(components, connections, ports_map)
    return component


def ring_single_bus_netlist(
    bend_radius=5,
    length_x=1,
    length_y=1,
    gap=0.2,
    bend90_factory=bend_circular,
    coupler90_factory=coupler90,
    straight_factory=waveguide,
    cpl_straight_factory=coupler_straight,
    wg_width=0.5,
):
    """
    .. code::

         BL--H1--BR
         |       |
         VL      VR
         |       |
        -CL==CS==CR-
    """
    assert_on_2nm_grid(gap)

    bend = bend90_factory(radius=bend_radius)
    cpl_bend = coupler90_factory(bend_radius=bend_radius, width=wg_width, gap=gap)
    cpl_straight = cpl_straight_factory(length=length_x, gap=gap, width=wg_width)
    h1 = straight_factory(length=length_x)
    v = straight_factory(length=length_y)

    components = {
        "CL": (cpl_bend, "mirror_y"),
        "CR": (cpl_bend, "None"),
        "BR": (bend, "mirror_x"),
        "BL": (bend, "R180"),
        "CS": (cpl_straight, "None"),
        "H1": (h1, "None"),
        "VL": (v, "R90"),
        "VR": (v, "R90"),
    }

    connections = [
        ("CL", "W0", "CS", "W0"),
        ("CS", "E0", "CR", "W0"),
        ("CR", "N0", "VR", "W0"),
        ("VR", "E0", "BR", "N0"),
        ("BR", "W0", "H1", "E0"),
        ("H1", "W0", "BL", "W0"),
        ("BL", "N0", "VL", "E0"),
        ("VL", "W0", "CL", "N0"),
    ]

    ports_map = {"W0": ("CL", "E0"), "E0": ("CR", "E0")}

    return components, connections, ports_map


if __name__ == "__main__":
    c = ring_single_bus(bend_radius=5.0, length_x=2, length_y=4, gap=0.2)
    # c = ring_single_bus_biased(bend_radius=5.0, length_x=2, length_y=4, gap=0.2)
    pp.show(c)
