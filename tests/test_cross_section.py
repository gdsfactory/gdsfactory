from __future__ import annotations

from functools import partial
from typing import Any

import jsondiff

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_waveguide_setting() -> None:
    x = gf.cross_section.cross_section(width=2)
    assert x.width == 2


def test_settings_different() -> None:
    strip1 = gf.cross_section.strip()
    strip2 = gf.cross_section.strip(layer=(2, 0))
    assert strip1 != strip2


def test_transition_names() -> None:
    layer = (1, 0)
    s1 = gf.Section(width=5, layer=layer, port_names=("o1", "o2"), name="core")
    s2 = gf.Section(width=50, layer=layer, port_names=("o1", "o2"), name="core")

    xs1 = gf.CrossSection(sections=(s1,))
    xs2 = gf.CrossSection(sections=(s2,))
    trans12 = gf.path.transition(
        cross_section1=xs1, cross_section2=xs2, width_type="linear"
    )
    trans21 = gf.path.transition(
        cross_section1=xs2, cross_section2=xs1, width_type="linear"
    )

    WG4Path = gf.Path()
    WG4Path.append(gf.path.straight(length=100, npoints=2))
    c1 = gf.path.extrude_transition(WG4Path, trans12)
    c2 = gf.path.extrude_transition(WG4Path, trans21)
    assert c1.name != c2.name


def test_copy() -> None:
    s = gf.Section(width=0.5, offset=0, layer=(3, 0), port_names=("in", "out"))
    x1 = gf.CrossSection(sections=(s,))
    x2 = x1.copy()
    d = jsondiff.diff(x1.model_dump(), x2.model_dump())
    assert len(d) == 0, d

    xs1 = gf.get_cross_section("metal_routing")
    xs2 = xs1.copy(width=2)
    assert xs2.name != xs1.name, f"{xs2.name} == {xs1.name}"

    xs1 = gf.get_cross_section("metal_routing")
    xs2 = xs1.copy(width=10)
    assert xs2.name == xs1.name, f"{xs2.name} != {xs1.name}"


def test_name() -> None:
    s = gf.cross_section.strip()
    assert s.name == "strip"


xc_sin = partial(
    gf.cross_section.cross_section,
    width=1.0,
    layer=(1, 0),
    cladding_layers=((1, 2), (1, 3)),
    cladding_offsets=(5, 10),
)

xc_sin_ec = partial(xc_sin, width=0.2)


@gf.cell
def demo_taper_cladding_offsets() -> gf.Component:
    taper_length = 10

    in_stub_length = 10
    out_stub_length = 10

    c = gf.Component()
    wg_in = c << gf.components.straight(length=in_stub_length, cross_section=xc_sin_ec)

    taper = c << gf.components.taper_cross_section_linear(
        length=taper_length, cross_section1=xc_sin_ec, cross_section2=xc_sin
    )

    wg_out = c << gf.components.straight(length=out_stub_length, cross_section=xc_sin)

    taper.connect("o1", wg_in.ports["o2"])
    wg_out.connect("o1", taper.ports["o2"])

    c.add_port("o1", port=wg_in.ports["o1"])
    c.add_port("o2", port=wg_out.ports["o2"])
    return c


def test_taper_cladding_offets() -> None:
    c = demo_taper_cladding_offsets()
    n = len(c.get_polygons()[LAYER.WG])
    assert n == 3, n


def test_is_cross_section_basic() -> None:
    def basic_xs(width: float = 1.0) -> gf.CrossSection:
        return gf.cross_section.cross_section(width=width, layer=(1, 0))

    assert gf.cross_section.is_cross_section("basic_xs", basic_xs)


def test_is_cross_section_subclass() -> None:
    class OtherCrossSection(gf.CrossSection):
        pass

    def cross_section(**kwargs: Any) -> OtherCrossSection:
        return OtherCrossSection(**kwargs)

    assert gf.cross_section.is_cross_section("cross_section", cross_section)


def test_is_cross_section_subclass_name_not_including_cross_section() -> None:
    class SubclassCrossSection(gf.CrossSection):
        pass

    def cross_section(**kwargs: Any) -> SubclassCrossSection:
        return SubclassCrossSection(**kwargs)

    assert gf.cross_section.is_cross_section("cross_section", cross_section)


def test_is_cross_section_partial() -> None:
    xs_partial = partial(gf.cross_section.cross_section, width=1.0, layer=(1, 0))
    assert gf.cross_section.is_cross_section("xs_partial", xs_partial)


def test_is_cross_section_invalid() -> None:
    def not_xs() -> None:
        pass

    assert not gf.cross_section.is_cross_section("not_xs", not_xs)
    assert not gf.cross_section.is_cross_section("len", len)


def test_is_cross_section_private() -> None:
    def _private_xs() -> gf.CrossSection:
        return gf.cross_section.cross_section(width=1.0, layer=(1, 0))

    assert not gf.cross_section.is_cross_section("_private_xs", _private_xs)


if __name__ == "__main__":
    # test_transition_names()
    # test_copy()
    test_taper_cladding_offets()
