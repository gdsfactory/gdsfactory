from typing import Callable, Dict, List, Tuple

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.components.waveguide import waveguide
from pp.drc import assert_on_2nm_grid
from pp.netlist_to_gds import netlist_to_component


@cell
def ring_double_bus(**kwargs) -> Component:
    """ Ring double bus

    Args:
        bend_radius=5
        length_x=1
        length_y=1
        gap=0.2
        coupler90_factory=coupler90
        straight_factory=waveguide
        cpl_straight_factory=coupler_straight
        wg_width=0.5


    .. code::

        -CTL==CTS==CTR-
         |          |
         VL         VR
         |          |
        -CBL==CBS==CBR-

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring_double_bus(gap=0.2, length_x=10, length_y=5, bend_radius=5)
      pp.plotgds(c)


    """
    components, connections, ports_map = ring_double_bus_netlist(**kwargs)
    component = netlist_to_component(components, connections, ports_map)
    return component


def ring_double_bus_netlist(
    bend_radius: float = 5.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
    gap: float = 0.2,
    coupler90_factory: Callable = coupler90,
    straight_factory: Callable = waveguide,
    cpl_straight_factory: Callable = coupler_straight,
    wg_width: float = 0.5,
) -> Tuple[
    Dict[str, Tuple[Component, str]],
    List[Tuple[str, str, str, str]],
    Dict[str, Tuple[str, str]],
]:
    """
    .. code::

         -CTL==CTS==CTR-
         |          |
         VL         VR
         |          |
        -CBL==CBS==CBR-
    """
    assert_on_2nm_grid(gap)

    _cpl_bend = coupler90_factory(bend_radius=bend_radius, width=wg_width, gap=gap)

    _cpl_straight = cpl_straight_factory(length=length_x, gap=gap, width=wg_width)
    _v = straight_factory(length=length_y)

    components = {
        "CBL": (_cpl_bend, "mirror_y"),
        "CBR": (_cpl_bend, "None"),
        "CTR": (_cpl_bend, "mirror_x"),
        "CTL": (_cpl_bend, "R180"),
        "CTS": (_cpl_straight, "None"),
        "CBS": (_cpl_straight, "None"),
        "VL": (_v, "R90"),
        "VR": (_v, "R90"),
    }

    connections = [
        ("CBL", "W0", "CBS", "W0"),
        ("CBS", "E0", "CBR", "W0"),
        ("CBL", "N0", "VL", "W0"),
        ("VL", "E0", "CTL", "N0"),
        ("CTL", "W0", "CTS", "W0"),
        ("CTS", "E0", "CTR", "W0"),
        ("CBR", "N0", "VR", "W0"),
        ("VR", "E0", "CTR", "N0"),
    ]

    ports_map = {
        "W0": ("CBL", "E0"),
        "E0": ("CBR", "E0"),
        "W1": ("CTL", "E0"),
        "E1": ("CTR", "E0"),
    }

    return components, connections, ports_map


if __name__ == "__main__":
    c = ring_double_bus(bend_radius=5.0, length_x=2, length_y=4, gap=0.2)
    pp.show(c)
