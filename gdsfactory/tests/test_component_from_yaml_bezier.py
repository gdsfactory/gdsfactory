import gdsfactory as gf
from gdsfactory.component import Component

yaml = """
name:
    test_component_yaml_without_cell

instances:
    mmi:
      component: mmi1x2
    bend:
      component: bend_s

connections:
    bend,o1: mmi,o2

"""


def test_component_from_yaml_without_cell() -> Component:
    """bezier does not have cell"""
    c = gf.read.from_yaml(yaml)
    assert c.name == "test_component_yaml_without_cell", c.name
    assert len(c.get_dependencies()) == 2, len(c.get_dependencies())
    assert len(c.ports) == 0, len(c.ports)
    return c


if __name__ == "__main__":
    c = test_component_from_yaml_without_cell()
    print(c.name)
    c.show()
