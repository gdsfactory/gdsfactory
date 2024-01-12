from __future__ import annotations

import numpy as np

import gdsfactory as gf


def get_custom_width_func(n: int = 5):
    def _width_func(t, y1, y2):
        return (y2 - y1) + np.cos(2 * np.pi * t * n)

    return _width_func


def test_transition_type_callable_multiple() -> None:
    """Ensure that the width_type serialization works properly"""
    width1 = 1.0
    width2 = 2.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)

    p = gf.path.straight(length=10, npoints=100)

    xt1 = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=get_custom_width_func(n=1)
    )
    xt2 = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=get_custom_width_func(n=5)
    )
    s1 = p.extrude(cross_section=xt1)
    s2 = p.extrude(cross_section=xt2)
    assert s1 is not s2


if __name__ == "__main__":
    test_transition_type_callable_multiple()

    width1 = 1.0
    width2 = 2.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)

    p = gf.path.straight(length=10, npoints=100)

    xt1 = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=get_custom_width_func(n=1)
    )
    xt2 = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=get_custom_width_func(n=5)
    )
    c = gf.Component()
    s1 = c << p.extrude(cross_section=xt1)
    s2 = c << p.extrude(cross_section=xt2)
    s2.movey(-10)

    c.show()
