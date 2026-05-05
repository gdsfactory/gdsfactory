from __future__ import annotations

import gdsfactory as gf


def test_path_transition_class() -> None:
    P = gf.path.straight(length=10, npoints=101)

    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("o1", "o2"))
    s1 = gf.Section(width=3, offset=0, layer=(3, 0))
    X1 = gf.CrossSection(sections=(s0, s1))
    X2 = gf.CrossSection(sections=(s0,))

    T = gf.path.transition(X1, X2)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_asymmetric_class() -> None:
    P = gf.path.straight(length=10, npoints=101)

    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("o1", "o2"))
    s1 = gf.Section(width=3, offset=0, layer=(3, 0))
    X1 = gf.CrossSection(sections=(s0, s1))
    X2 = gf.CrossSection(sections=(s0,))

    T = gf.path.transition_asymmetric(X1, X2)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_function() -> None:
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.cross_section(width=1)
    X2 = gf.cross_section.cross_section(width=3)
    T = gf.path.transition(X1, X2)
    P = gf.path.straight(length=10, npoints=101)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_asymmetric_function() -> None:
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.cross_section(width=1)
    X2 = gf.cross_section.cross_section(width=3)
    T = gf.path.transition_asymmetric(X1, X2)
    P = gf.path.straight(length=10, npoints=101)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_transition_ports() -> None:
    width1 = 0.5
    width2 = 1.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)
    xt = gf.path.transition(cross_section1=x1, cross_section2=x2, width_type="linear")
    path = gf.path.straight(length=5)
    c = gf.path.extrude_transition(path, xt)

    assert c.ports["o1"].width == width1, c.ports["o1"].width
    assert c.ports["o2"].width == width2, c.ports["o2"].width

    assert c.ports["o1"].width == width1, c.ports["o1"].width
    assert c.ports["o2"].width == width2, c.ports["o2"].width


def test_transition_asymmetric_ports() -> None:
    width1 = 0.5
    width2 = 1.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)
    xt = gf.path.transition_asymmetric(
        cross_section1=x1, cross_section2=x2, width_type1="linear"
    )
    path = gf.path.straight(length=5, npoints=10)
    c = gf.path.extrude_transition(path, xt)

    assert c.ports["o1"].width == width1, c.ports["o1"].width
    assert c.ports["o2"].width == width2, c.ports["o2"].width

    assert c.ports["o1"].width == width1, c.ports["o1"].width
    assert c.ports["o2"].width == width2, c.ports["o2"].width


def test_taper_cross_section_round_tripped_layer_spec() -> None:
    xs1 = gf.cross_section.strip(width=0.5)
    xs2 = gf.cross_section.strip(width=1.0)

    t1 = gf.components.taper_cross_section(xs1, xs2, length=10, linear=True)
    xs1_rt = gf.get_cross_section(t1.ports["o1"].cross_section)
    assert xs1_rt.sections[0].layer == xs1.sections[0].layer

    t2 = gf.components.taper_cross_section(xs1_rt, xs2, length=10, linear=True)
    assert any(t2.get_polygons(merge=False).values())
