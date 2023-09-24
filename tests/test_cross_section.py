from __future__ import annotations

from functools import partial

import gdsfactory as gf


def test_waveguide_setting() -> None:
    x = gf.cross_section.cross_section(width=2)
    assert x.width == 2


def test_settings_different() -> None:
    strip1 = gf.cross_section.strip()
    strip2 = gf.cross_section.strip(layer=(2, 0))
    assert strip1 != strip2


def test_transition_names() -> None:
    layer = (1, 0)
    xs1 = gf.CrossSection(width=5, layer=layer, port_names=("o1", "o2"))
    xs2 = gf.CrossSection(width=50, layer=layer, port_names=("o1", "o2"))

    trans12 = gf.path.transition(
        cross_section1=xs1, cross_section2=xs2, width_type="linear"
    )
    trans21 = gf.path.transition(
        cross_section1=xs2, cross_section2=xs1, width_type="linear"
    )

    WG4Path = gf.Path()
    WG4Path.append(gf.path.straight(length=100, npoints=2))
    c1 = gf.path.extrude(WG4Path, cross_section=trans12)
    c2 = gf.path.extrude(WG4Path, cross_section=trans21)
    assert c1.name != c2.name


def test_cross_section_autoname() -> None:
    x = gf.CrossSection(width=0.5)
    assert x.name


if __name__ == "__main__":
    pin = partial(
        gf.cross_section.strip,
        layer=(2, 0),
        sections=(
            gf.Section(layer=(21, 0), width=2, offset=+2),
            gf.Section(layer=(20, 0), width=2, offset=-2),
        ),
    )
    c = gf.components.straight(cross_section=pin)
