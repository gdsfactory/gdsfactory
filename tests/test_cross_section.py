from __future__ import annotations

from functools import partial

import pytest
from hypothesis import given
from hypothesis import strategies as st

import gdsfactory as gf
from gdsfactory._kcl import temporary_kcl
from gdsfactory.gpdk import LAYER


@given(
    width=st.floats(
        min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
    )
)
def test_waveguide_setting(width: float) -> None:
    with temporary_kcl("width_snapping") as kcl:
        x = gf.cross_section.cross_section(width=width, kcl=kcl)
        assert x.width == kcl.to_um(kcl.to_dbu(width / 2) - kcl.to_dbu(-width / 2))


def test_settings_different() -> None:
    strip1 = gf.cross_section.strip()
    strip2 = gf.cross_section.strip(layer=(902, 0))
    assert strip1 != strip2


def test_transition_names() -> None:
    layer = (1, 0)
    s1 = (layer, -2.5, 2.5)
    s2 = (layer, -25.0, 25.0)

    xs1 = gf.cross_section.strip(width=None, sections=(s1,))
    xs2 = gf.cross_section.strip(width=None, sections=(s2,))
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


def test_transition_asymmetric_names() -> None:
    layer = (1, 0)
    s1 = (layer, -2.5, 2.5)
    s2 = (layer, -25.0, 25.0)

    xs1 = gf.cross_section.strip(width=None, sections=(s1,))
    xs2 = gf.cross_section.strip(width=None, sections=(s2,))
    trans12 = gf.path.transition_asymmetric(
        cross_section1=xs1, cross_section2=xs2, width_type1="linear", width_type2="sine"
    )
    trans21 = gf.path.transition_asymmetric(
        cross_section1=xs2, cross_section2=xs1, width_type1="linear", width_type2="sine"
    )

    WG4Path = gf.Path()
    WG4Path.append(gf.path.straight(length=100, npoints=2))
    c1 = gf.path.extrude_transition(WG4Path, trans12)
    c2 = gf.path.extrude_transition(WG4Path, trans21)
    assert c1.name != c2.name


def test_replace_width() -> None:
    xs = gf.cross_section.rib()
    wider = gf.cross_section.with_width(xs, 1.1)
    assert wider.width == 1.1
    assert wider.get_sections()[1:] == xs.get_sections()[1:]
    assert gf.cross_section.with_width(xs, xs.width) is xs


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


def test_cross_section_aliases() -> None:
    import kfactory as kf

    assert gf.SymmetricCrossSection is kf.DCrossSection
    assert gf.AsymmetricCrossSection is kf.DAsymmetricCrossSection
    assert gf.CrossSection == kf.DCrossSection | kf.DAsymmetricCrossSection
    assert not hasattr(gf, "Section")


def test_is_cross_section_partial() -> None:
    xs_partial = partial(gf.cross_section.cross_section, width=1.0, layer=(1, 0))
    assert gf.cross_section.is_cross_section("xs_partial", xs_partial)


def test_is_cross_section_invalid() -> None:
    def not_xs() -> None:
        pass

    assert not gf.cross_section.is_cross_section("not_xs", not_xs)
    assert not gf.cross_section.is_cross_section("len", len)


@pytest.mark.parametrize("width", [0, -1, 0.0001])
def test_section_requires_positive_snapped_width(width: float) -> None:
    with pytest.raises(ValueError):
        gf.cross_section.cross_section(width=width)


def test_variable_width_and_offset_belong_to_extrusion() -> None:
    xs = gf.cross_section.cross_section(
        width=0.5, layer="WG", cladding_layers=("SLAB90",), cladding_offsets=(1.0,)
    )
    path = gf.path.straight(10.0, npoints=101)
    component = path.extrude(
        xs,
        width_function={0: lambda t: 0.5 + t, 1: lambda t: 2.5 + t},
        offset_function=lambda t: 0.2 * t,
    )
    assert component.ports["o1"].width == 0.5
    assert component.ports["o2"].width == 1.5
    assert component.ports["o2"].y == pytest.approx(0)
    main, slab = component.ports["o2"].cross_section.get_sections()
    assert (main.section_min + main.section_max) / 2 == pytest.approx(0.2)
    assert slab.width == 3.5
    assert xs.width == 0.5
    assert not hasattr(xs, "width_function")


def test_is_cross_section_private() -> None:
    def _private_xs() -> gf.CrossSection:
        return gf.cross_section.cross_section(width=1.0, layer=(1, 0))

    assert not gf.cross_section.is_cross_section("_private_xs", _private_xs)


def test_taper_cross_section_instance_matches_name() -> None:
    """Taper must honor width overrides when cross_section is a CrossSection.

    Passing a CrossSection instance used to drop the taper's width overrides
    (and reuse one geometry for both ports), giving a different result than the
    equivalent string spec and poisoning the cell cache (#4588).
    """
    from gdsfactory.cross_section import CrossSection, xsection

    @xsection
    def _xs_4588(width: float = 0.5) -> CrossSection:
        return gf.cross_section.strip(
            width=None,
            sections=(((1, 0), -(width / 2), width / 2),),
        )

    def _from_spec() -> gf.Component:
        return gf.components.taper(
            width2=10, cross_section=gf.get_cross_section("_xs_4588")
        )

    def _from_str() -> gf.Component:
        return gf.components.taper(width2=10, cross_section="_xs_4588")

    pdk = gf.get_active_pdk()
    previous = pdk.cross_sections.get("_xs_4588")
    pdk.cross_sections["_xs_4588"] = _xs_4588
    try:
        # build the instance- and string-based tapers in both orders, so the
        # test catches cache poisoning regardless of which one is built first
        for builders in ((_from_spec, _from_str), (_from_str, _from_spec)):
            gf.clear_cache()
            for build in builders:
                taper = build()
                assert taper.ports["o1"].width == 0.5
                assert taper.ports["o2"].width == 10.0
    finally:
        if previous is None:
            pdk.cross_sections.pop("_xs_4588", None)
        else:
            pdk.cross_sections["_xs_4588"] = previous
