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
                mmi_short,2: mmi_long,3

    ports:
        2: mmi_short,1
        1: mmi_long,1
    """

    c = gf.component_from_yaml(yaml)
    return c


if __name__ == "__main__":
    c = test_netlist_yaml()
    c.show()
