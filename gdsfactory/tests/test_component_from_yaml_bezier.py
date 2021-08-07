import gdsfactory as gf
from gdsfactory.component import Component

yaml = """
name:
    test_component_from_yaml_without_cell

instances:
    mmi:
      component: mmi1x2
    bend:
      component: bezier

connections:
    bend,0: mmi,E1

"""


def test_component_from_yaml_without_cell() -> Component:
    """bezier does not have cell"""
    c = gf.component_from_yaml(yaml)
    assert c.name == "test_component_from_yaml_without_cell"
    assert len(c.get_dependencies()) == 2
    assert len(c.ports) == 0
    return c


if __name__ == "__main__":
    c = test_component_from_yaml_without_cell()
    print(c.name)
    c.show()
