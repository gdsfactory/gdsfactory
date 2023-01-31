###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2022 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################
from __future__ import annotations

from gdsfactory.read import from_yaml
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.routing.all_angle import get_bundle_all_angle


@cell
def demo_aar() -> Component:
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
            rotation: 190
            x: 100
            y: 100

    routes:
        optical:
            routing_strategy: get_bundle_all_angle
            links:
                mmi_short,o2: mmi_long,o3
                mmi_short,o3: mmi_long,o2

    ports:
        o2: mmi_short,o1
        o1: mmi_long,o1
    """

    return from_yaml(yaml)


if __name__ == "__main__":
    from gdsfactory.routing.factories import routing_strategy

    routing_strategy["get_bundle_all_angle"] = get_bundle_all_angle
    c = demo_aar()
    c.show(show_ports=True)
