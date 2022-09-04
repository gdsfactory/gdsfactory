"""FIXME."""

import gdsfactory as gf

yaml = """
instances:
    mmi1:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi2:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

    straight:
        component: straight

placements:
    mmi2:
        x: 100
        mirror: True

    straight:
        x: 40
        y: 40

routes:
    route_name1:
        links:
            mmi1,o3: mmi2,o3
    route_name2:
        links:
            mmi1,o2: straight,o1
    route_name3:
        links:
            mmi2,o2: straight,o2

ports:
    o1: mmi2,o1
    o2: mmi2,o1
"""

if __name__ == "__main__":

    mzi = gf.read.from_yaml(yaml)
    mzi.show()
    n = mzi.get_netlist()
    # mzi.plot()
