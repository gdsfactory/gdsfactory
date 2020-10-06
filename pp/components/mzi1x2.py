from typing import Callable
import pp

from pp.components import bend_circular
from pp.components import wg_heater_connected as waveguide_heater
from pp.components import waveguide
from pp.components import mmi1x2
from pp.components.mzi2x2 import mzi_arm
from pp.netlist_to_gds import netlist_to_component
from pp.routing import route_elec_ports_to_side
from pp.port import select_electrical_ports

from pp.components.extension import line
from pp.component import Component


@pp.autoname
def mzi1x2(
    L0: float = 0.1,
    DL: float = 9.0,
    L2: float = 10.0,
    bend_radius: float = 10.0,
    bend90_factory: Callable = bend_circular,
    straight_heater_factory: Callable = waveguide_heater,
    straight_factory: Callable = waveguide,
    coupler_factory: Callable = mmi1x2,
    with_elec_connections: bool = False,
) -> Component:
    """ Mzi 1x2

    Args:
        L0: vertical length for both and top arms
        DL: bottom arm extra length
        L2: L_top horizontal length
        bend_radius: 10.0
        bend90_factory: bend_circular
        straight_heater_factory: waveguide_heater or waveguide
        straight_factory: waveguide
        coupler_factory: coupler

    .. code::

             __L2__
            |      |
            L0     L0
            |      |
          --|      |--
            |      |
            L0     L0
            |      |
            DL     DL
            |      |
            |__L2__|


             top_arm
        -CP1=       =CP2-
             bot_arm


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi1x2(L0=0.1, DL=0, L2=10)
      pp.plotgds(c)

    """
    if not with_elec_connections:
        straight_heater_factory = straight_factory

    cpl = pp.call_if_func(coupler_factory)

    arm_defaults = {
        "L_top": L2,
        "bend_radius": bend_radius,
        "bend90_factory": bend90_factory,
        "straight_heater_factory": straight_heater_factory,
        "straight_factory": straight_factory,
        "with_elec_connections": with_elec_connections,
    }

    arm_top = mzi_arm(L0=L0, **arm_defaults)
    arm_bot = mzi_arm(L0=L0 + DL, **arm_defaults)

    components = {
        "CP1": (cpl, "None"),
        "CP2": (cpl, "mirror_y"),
        "arm_top": (arm_top, "None"),
        "arm_bot": (arm_bot, "mirror_x"),
    }

    connections = [
        # Bottom arm
        ("CP1", "E0", "arm_bot", "W0"),
        ("arm_bot", "E0", "CP2", "E0"),
        # Top arm
        ("CP1", "E1", "arm_top", "W0"),
        ("arm_top", "E0", "CP2", "E0"),
    ]

    if with_elec_connections:
        ports_map = {
            "W0": ("CP1", "W0"),
            "E0": ("CP2", "W0"),
            "E_TOP_0": ("arm_top", "E_0"),
            "E_TOP_1": ("arm_top", "E_1"),
            "E_TOP_2": ("arm_top", "E_2"),
            "E_TOP_3": ("arm_top", "E_3"),
            "E_BOT_0": ("arm_bot", "E_0"),
            "E_BOT_1": ("arm_bot", "E_1"),
            "E_BOT_2": ("arm_bot", "E_2"),
            "E_BOT_3": ("arm_bot", "E_3"),
        }

        component = netlist_to_component(components, connections, ports_map)
        # Need to connect common ground and redefine electrical ports

        ports = component.ports
        y_elec = ports["E_TOP_0"].y
        for ls, le in [
            ("E_BOT_0", "E_BOT_1"),
            ("E_TOP_0", "E_TOP_1"),
            ("E_BOT_2", "E_TOP_2"),
        ]:
            component.add_polygon(line(ports[ls], ports[le]), layer=ports[ls].layer)

        # Add GND
        ("E_BOT_2", "E_TOP_2")
        component.add_port(
            name="GND",
            midpoint=0.5 * (ports["E_BOT_2"].midpoint + ports["E_TOP_2"].midpoint),
            orientation=180,
            width=ports["E_BOT_2"].width,
            layer=ports["E_BOT_2"].layer,
        )

        component.ports["E_TOP_3"].orientation = 0
        component.ports["E_BOT_3"].orientation = 0

        # Remove the eletrical ports that we have just used internally
        for lbl in ["E_BOT_0", "E_BOT_1", "E_TOP_0", "E_TOP_1", "E_BOT_2", "E_TOP_2"]:
            component.ports.pop(lbl)

        # Reroute electrical ports
        _e_ports = select_electrical_ports(component)
        conn, e_ports = route_elec_ports_to_side(_e_ports, side="north", y=y_elec)

        for c in conn:
            component.add(c)

        for p in e_ports:
            component.ports[p.name] = p

        # Create nice electrical port names
        component.ports["HT1"] = component.ports["E_TOP_3"]
        component.ports.pop("E_TOP_3")

        component.ports["HT2"] = component.ports["E_BOT_3"]
        component.ports.pop("E_BOT_3")

    else:
        ports_map = {"W0": ("CP1", "W0"), "E0": ("CP2", "W0")}
        component = netlist_to_component(components, connections, ports_map)

    return component


if __name__ == "__main__":
    import pp

    c = mzi1x2(coupler_factory=mmi1x2, with_elec_connections=False)
    # print(c.ports)
    pp.show(c)
    # print(c.get_settings())
