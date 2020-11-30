""" needs some fix
maybe add routing
"""

import pp

tab = " " * 12

ports = "\n".join([f"{tab}w,E{i}: d,W{i}" for i in range(4)])

yaml = f"""
instances:
    w:
        component: waveguide_array
    d:
        component: nxn
        settings:
            west: 4

placements:
    d:
        x: w,E0
        dx: 50

routes:
    optical:
        factory: optical
        links:
{ports}

"""


if __name__ == "__main__":
    print(yaml)
    c = pp.component_from_yaml(yaml)
    pp.show(c)
