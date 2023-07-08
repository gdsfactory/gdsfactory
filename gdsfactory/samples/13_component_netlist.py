from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def netlist_yaml() -> Component:
    """Test netlist yaml.

    .. code::

        arm_top
         _____
        |     |
    CP1=       =CP2=
        |_____|

         arm_bot

    """
    yaml = """
    instances:
        mmi_long:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 10
        mmi_short:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 5

    placements:
        mmi_long:
            rotation: 180
            x: 100
            y: 100

    routes:
        optical:
            links:
                mmi_short,o2: mmi_long,o3

    ports:
        o2: mmi_short,o1
        o1: mmi_long,o1
    """

    return gf.read.from_yaml(yaml)


def test_netlist_yaml_sample() -> None:
    assert netlist_yaml()


if __name__ == "__main__":
    c = netlist_yaml()
    c.show(show_ports=True)
