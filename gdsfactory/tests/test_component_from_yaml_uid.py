import gdsfactory as gf

mirror_port = """
name: mirror_port

instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: o1
        x: 20
        y: 10
        mirror: True

ports:
    o1: mmi_long,o3
    o2: mmi_long,o2
    o3: mmi_long,o1

info:
    wavelength: 1.55
    docstring: MMI with a mirror port

"""


def test_components_ports() -> None:
    """Make sure the same component gets the same UID"""
    c1 = gf.read.from_yaml(mirror_port)
    c2 = gf.read.from_yaml(mirror_port)
    # print(c1.uid)
    # print(c2.uid)
    assert c1.uid == c2.uid


if __name__ == "__main__":
    c1 = gf.read.from_yaml(mirror_port)
    c2 = gf.read.from_yaml(mirror_port)
    print(c1.uid)
    print(c2.uid)
    assert c1.uid == c2.uid
    c2.show()
