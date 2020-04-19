"""
Sometimes, when a component is mostly composed of sub-components adjacent to each other, it can be easier to define the component by:

- subcomponents
- connections
- exposed ports in the new components

This can be done using a netlist based approach where these 3 parts are defined:

- components: a dictionnary of {component reference name: (component, transform)}
- connections: a list of (component ref name 1, port name A, component ref name 2, port name B)
- ports_map: a dictionnary of which ports are being exposed together with their new name {port_name: (component ref name, port name)}

The code below illustrates how a simple MZI can be formed using this method.

"""


import pp
from pp.netlist_to_gds import netlist_to_component


@pp.autoname
def test_simple_mzi2x2(
    coupler_length=20.147, arm_length=40, gap=0.234, waveguide_factory=pp.c.waveguide
):
    """
       arm_top
        _____
       |     |
   CP1=       =CP2=
       |_____|

        arm_bot
   """

    # Define the components to use
    coupler = pp.c.coupler(length=coupler_length, gap=gap)
    arm = waveguide_factory(length=arm_length)

    # Create component references and give them unique names {name : (component, transform)}
    # Transform can be either "None", "mirror_x", "mirror_y", "R90", "R180", "R270"

    components = {
        "CP1": (coupler, "None"),
        "CP2": (coupler, "None"),
        "arm_top": (arm, "None"),
        "arm_bot": (arm, "mirror_x"),
    }

    # Provide how components are connected (component 1, port A, component 2, port B)
    # means that port B from component 2 is positioned at port A from component 1

    connections = [
        ## Top arm
        ("CP1", "E1", "arm_top", "W0"),
        ("arm_top", "E0", "CP2", "W1"),
        ## Bottom arm
        ("CP1", "E0", "arm_bot", "W0"),
        ("arm_bot", "E0", "CP2", "W0"),
    ]

    # Create the ports for this component, choosing which ports from the subcomponents should be exposed
    # {port name: (subcomponent, subcomponent port name)}

    ports_map = {
        "W0": ("CP1", "W0"),
        "W1": ("CP1", "W1"),
        "E0": ("CP2", "E0"),
        "E1": ("CP2", "E1"),
    }

    component = netlist_to_component(components, connections, ports_map)
    return component


def test_mzi():
    c = test_simple_mzi2x2()
    assert c
    pp.show(c)


def test_mzi_heated():
    c = test_simple_mzi2x2(waveguide_factory=pp.c.waveguide_heater)
    assert c
    pp.show(c)


if __name__ == "__main__":
    test_mzi_heated()
