from __future__ import annotations

from gdsfactory.read.from_yaml import from_yaml

yaml = """
name: mmis

instances:
    mmi1:
        component: mmi1x2

    mmi2:
        component: mmi1x2

placements:
    mmi2:
        port: o2
        x: mmi1,o2
        dy: -0.625
        mirror: True

"""


if __name__ == "__main__":
    c = from_yaml(yaml)
    c.show()
