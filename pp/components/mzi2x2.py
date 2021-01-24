from typing import Callable

from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.component_sequence import component_sequence
from pp.components.coupler import coupler
from pp.components.extension import line
from pp.components.waveguide import waveguide
from pp.components.waveguide_heater import wg_heater_connected
from pp.netlist_to_gds import netlist_to_component
from pp.port import select_electrical_ports
from pp.routing import route_elec_ports_to_side


@cell
def mzi_arm(
    L0: float = 60.0,
    DL: float = 0.0,
    L_top: float = 10.0,
    bend_radius: float = 10.0,
    bend90_factory: Callable = bend_circular,
    waveguide_heater_function: Callable = wg_heater_connected,
    waveguide_function: Callable = waveguide,
    with_elec_connections: bool = True,
) -> Component:
    """

    Args:
        L0: vertical length with heater
        DL: extra vertical length without heater (delat_length=2*DL)
        L_top: 10.0, horizontal length
        bend_radius: 10.0
        bend90_factory: bend_circular
        waveguide_heater_function: wg_heater_connected
        waveguide_function: waveguide

    ::

         L_top
        |     |
        DL    DL
        |     |
        L0    L0
        |     |
       -|     |-

        B2-Sh1-B3
        |     |
        Sv1   Sv2
        |     |
        H1    H2
        |     |
       -B1    B4-


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi_arm()
      pp.plotgds(c)


    """
    if not with_elec_connections:
        waveguide_heater_function = waveguide_function

    _bend = bend90_factory(radius=bend_radius)

    straight_vheater = waveguide_heater_function(length=L0)
    straight_h = waveguide_function(length=L_top)
    straight_v = waveguide_function(length=DL) if DL > 0 else None

    string_to_device_in_out_ports = {
        "A": (_bend, "W0", "N0"),
        "B": (_bend, "N0", "W0"),
        "H": (straight_vheater, "W0", "E0"),
        "Sh": (straight_h, "W0", "E0"),
        "Sv": (straight_v, "W0", "E0"),
    }

    sequence = ["A", "Sv", "H", "B", "Sh", "B", "H", "Sv", "A"]

    if with_elec_connections:
        ports_map = {
            "E_0": ("H2", "E_1"),
            "E_1": ("H1", "E_0"),
            "E_2": ("H1", "E_1"),
            "E_3": ("H2", "E_0"),
        }
    else:
        ports_map = {}

    component = component_sequence(
        sequence,
        string_to_device_in_out_ports,
        ports_map,
        input_port_name="W0",
        output_port_name="E0",
    )
    return component


@cell
def mzi2x2(
    CL_1: float = 20.147,
    L0: float = 60.0,
    DL: float = 7.38,
    L2: float = 10.0,
    gap: float = 0.234,
    bend_radius: float = 10.0,
    bend90_factory: Callable = bend_circular,
    waveguide_heater_function: Callable = wg_heater_connected,
    waveguide_function: Callable = waveguide,
    coupler_function: Callable = coupler,
    with_elec_connections: bool = False,
) -> Component:
    """Mzi 2x2

    Args:
        CL_1: coupler length
        L0: vertical length for both and top arms
        DL: bottom arm extra length
        L2: L_top horizontal length
        gap: 0.235
        bend_radius: 10.0
        bend90_factory: bend_circular
        waveguide_heater_function: wg_heater_connected or waveguide
        waveguide_function: waveguide
        coupler_function: coupler


    .. code::

         __L2__
        |      |
        L0     L0
        |      |
      ==|      |==
        |      |
        L0     L0
        |      |
        DL     DL
        |      |
        |__L2__|


    .. code::

               top_arm
        ==CL_1=       =CL_1===
               bot_arm


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi2x2(CL_1=10., gap=0.2)
      pp.plotgds(c)

    """
    if not with_elec_connections:
        waveguide_heater_function = waveguide_function

    cpl = coupler_function(length=CL_1, gap=gap)

    arm_defaults = {
        "L_top": L2,
        "bend_radius": bend_radius,
        "bend90_factory": bend90_factory,
        "waveguide_heater_function": waveguide_heater_function,
        "waveguide_function": waveguide_function,
        "with_elec_connections": with_elec_connections,
    }

    arm_top = mzi_arm(L0=L0, **arm_defaults)
    arm_bot = mzi_arm(L0=L0, DL=DL, **arm_defaults)

    components = {
        "CP1": (cpl, "None"),
        "CP2": (cpl, "None"),
        "arm_top": (arm_top, "None"),
        "arm_bot": (arm_bot, "mirror_x"),
    }

    connections = [
        # Top arm
        ("CP1", "E1", "arm_top", "W0"),
        ("arm_top", "E0", "CP2", "W1"),
        # Bottom arm
        ("CP1", "E0", "arm_bot", "W0"),
        ("arm_bot", "E0", "CP2", "W0"),
    ]

    if with_elec_connections:

        ports_map = {
            "W0": ("CP1", "W0"),
            "W1": ("CP1", "W1"),
            "E0": ("CP2", "E0"),
            "E1": ("CP2", "E1"),
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
        component.add_port(
            name="GND",
            midpoint=0.5 * (ports["E_BOT_2"].midpoint + ports["E_TOP_2"].midpoint),
            orientation=180,
            width=ports["E_BOT_2"].width,
            layer=ports["E_BOT_2"].layer,
            port_type="dc",
        )

        component.ports["E_TOP_3"].orientation = 0
        component.ports["E_BOT_3"].orientation = 0

        # Remove the eletrical ports that we have just used internally
        for lbl in ["E_BOT_0", "E_BOT_1", "E_TOP_0", "E_TOP_1", "E_BOT_2", "E_TOP_2"]:
            component.ports.pop(lbl)

        # Reroute electrical ports
        _e_ports = select_electrical_ports(component)
        routes, e_ports = route_elec_ports_to_side(_e_ports, side="north", y=y_elec)

        for route in routes:
            component.add(route["references"])

        for p in e_ports:
            component.ports[p.name] = p

        # Create nice electrical port names
        component.ports["HT1"] = component.ports["E_TOP_3"]
        component.ports.pop("E_TOP_3")

        component.ports["HT2"] = component.ports["E_BOT_3"]
        component.ports.pop("E_BOT_3")

        # Make sure each port knows its name
        for k, p in component.ports.items():
            p.name = k

    else:
        ports_map = {
            "W0": ("CP1", "W0"),
            "W1": ("CP1", "W1"),
            "E0": ("CP2", "E0"),
            "E1": ("CP2", "E1"),
        }

        component = netlist_to_component(components, connections, ports_map)

    return component


def get_mzi_delta_length(m, neff=2.4, wavelength=1.55):
    """ m*wavelength = neff * delta_length """
    return m * wavelength / neff


if __name__ == "__main__":
    import pp

    # print(get_mzi_delta_length(m=15))
    # print(get_mzi_delta_length(m=150))
    # for p in c.ports.values():
    #     print(p.port_type)
    # c = mzi_arm(DL=100)
    # c = mzi2x2(waveguide_heater_function=wg_heater_connected, with_elec_connections=True)
    # pp.write_gds(c, "mzi.gds")
    # print(c)
    # print(hash(frozenset(c.settings.items())))
    # print(hash(c))

    c = mzi2x2(with_elec_connections=True)
    cc = pp.add_pins(c)
    pp.show(cc)
