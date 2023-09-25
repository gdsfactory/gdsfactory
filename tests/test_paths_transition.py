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


def test_path_transition_function() -> None:
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.cross_section(width=1)
    X2 = gf.cross_section.cross_section(width=3)
    T = gf.path.transition(X1, X2)
    P = gf.path.straight(length=10, npoints=101)
    c = gf.path.extrude_transition(P, T)
    assert c
