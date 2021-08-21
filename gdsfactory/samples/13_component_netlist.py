import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def test_netlist_yaml() -> Component:
    """

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

    c = gf.component_from_yaml(yaml)
    return c


if __name__ == "__main__":
    c = test_netlist_yaml()
    c.show()
