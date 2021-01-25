import pp
from pp.component import Component

netlist = """
instances:
    CP1:
      component: mmi1x2
      settings:
          width_mmi: 4.5
          length_mmi: 10
    CP2:
        component: mmi1x2
        settings:
            width_mmi: 4.5
            length_mmi: 5
    arm_top:
        component: mzi_arm
    arm_bot:
        component: mzi_arm

placements:
    arm_bot:
        mirror: True
ports:
    W0: CP1,W0
    E0: CP2,W0

connections:
    arm_bot,W0: CP1,E0
    arm_top,W0: CP1,E1
    CP2,E0: arm_bot,E0
    CP2,E1: arm_top,E0
"""


def test_mzi() -> Component:
    return pp.component_from_yaml(netlist)


if __name__ == "__main__":
    c = test_mzi()
    pp.show(c)
    pp.plotgds(c)
