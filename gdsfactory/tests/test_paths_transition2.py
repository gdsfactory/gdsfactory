import pytest
from phidl.path import transition

import gdsfactory as gf


def test_transition_unamed_fails():
    """raises error when transitioning un-named cross_sections"""
    with pytest.raises(ValueError):

        path = gf.Path()
        path.append(gf.path.arc(radius=10, angle=90))
        path.append(gf.path.straight(length=10))
        path.append(gf.path.euler(radius=3, angle=-90))
        path.append(gf.path.straight(length=40))
        path.append(gf.path.arc(radius=8, angle=-45))
        path.append(gf.path.straight(length=10))
        path.append(gf.path.arc(radius=8, angle=45))
        path.append(gf.path.straight(length=10))

        X = gf.CrossSection()
        X.add(width=1, offset=0, layer=0)

        x2 = gf.CrossSection()
        x2.add(width=2, offset=0, layer=0)

        transition(X, x2)


def test_transition_ports():
    width1 = 0.5
    width2 = 1.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)
    xt = gf.path.transition(cross_section1=x1, cross_section2=x2, width_type="linear")
    path = gf.path.straight(length=5)
    c = gf.path.extrude(path, xt)
    assert c.ports["o1"].cross_section.info["width"] == width1
    assert c.ports["o2"].cross_section.info["width"] == width2


if __name__ == "__main__":
    # test_transition_ports()
    # x1 = gf.cross_section.strip(width=0.5)
    # x2 = gf.cross_section.strip(width=1.0)
    # xt = gf.path.transition(cross_section1=x1, cross_section2=x2, width_type="linear")
    # path = gf.path.straight(length=5)
    # c = gf.path.extrude(path, xt)
    test_transition_ports()
