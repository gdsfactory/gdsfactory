import pp

yaml = """

instances:
    mmi1:
        component: mmi1x2

    mmi2:
        component: mmi1x2

placements:
    mmi2:
        port: 'E0'
        x: mmi1,E0
        dy: -0.625
        mirror: True

"""


if __name__ == "__main__":
    c = pp.component_from_yaml(yaml)
    pp.show(c)
