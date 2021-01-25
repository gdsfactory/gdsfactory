import pp

yaml_fail = """
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
    mmi_short:
        port: W0
        x: mmi_long,E1
        y: mmi_long,E1
    mmi_long:
        port: W0
        x: mmi_short,E1
        y: mmi_short,E1
        dx : 10
        dy: 20
"""

yaml_pass = """
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
    mmi_short:
        port: W0
        x: 0
        y: 0
    mmi_long:
        port: W0
        x: mmi_short,E1
        y: mmi_short,E1
        dx : 10
        dy: 20
"""


# def test_circular_import_fail():
#     with pytest.raises(ValueError):
#         pp.component_from_yaml(yaml_fail)


def test_circular_import_pass() -> None:
    pp.component_from_yaml(yaml_pass)


if __name__ == "__main__":
    # c = test_circular_import_pass()
    # c = test_circular_import_fail()
    c = pp.component_from_yaml(yaml_fail)
    pp.show(c)
