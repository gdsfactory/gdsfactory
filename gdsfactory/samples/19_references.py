from __future__ import annotations

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
