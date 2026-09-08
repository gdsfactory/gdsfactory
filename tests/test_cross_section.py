from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pydantic import ValidationError

import gdsfactory as gf
from gdsfactory.gpdk import LAYER


@given(
    width=st.floats(
        min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
    )
)
def test_waveguide_setting(width: float) -> None:
    x = gf.cross_section.cross_section(width=width)
    assert x.width == pytest.approx(width, abs=2 * gf.kcl.dbu)


def test_settings_different() -> None:
    strip1 = gf.cross_section.strip()
    strip2 = gf.cross_section.strip(layer=(2, 0))
    assert strip1 != strip2


def test_transition_names() -> None:
    layer = (1, 0)
    xs1 = gf.cross_section.cross_section(
        width=5, layer=layer, name="transition_names_xs1"
    )
    xs2 = gf.cross_section.cross_section(
        width=50, layer=layer, name="transition_names_xs2"
    )
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
    xs1 = gf.cross_section.cross_section(
        width=5, layer=layer, name="transition_asymmetric_names_xs1"
    )
    xs2 = gf.cross_section.cross_section(
        width=50, layer=layer, name="transition_asymmetric_names_xs2"
    )
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


def test_copy() -> None:
    x1 = gf.cross_section.cross_section(width=0.5, layer=(3, 0))
    x2 = gf.cross_section.copy_cross_section(x1)
    assert x1 is x2

    native_wide = gf.cross_section.copy_cross_section(x1, width=2)
    assert native_wide.width == pytest.approx(2, abs=2 * gf.kcl.dbu)

    native = gf.get_cross_section("metal_routing")
    assert isinstance(native, gf.DCrossSection)


def test_name() -> None:
    s = gf.cross_section.strip()
    assert s.name == "strip"


def test_cross_section_returns_native_symmetric_profile() -> None:
    xs = gf.cross_section.cross_section(
        width=0.61,
        layer=(101, 0),
        radius=None,
        radius_min=None,
        name="native_symmetric_profile",
    )

    assert isinstance(xs, gf.DCrossSection)
    assert xs.name == "native_symmetric_profile"
    assert xs.width == pytest.approx(0.61, abs=2 * gf.kcl.dbu)


def test_cross_section_returns_native_asymmetric_profile() -> None:
    xs = gf.cross_section.cross_section(
        width=0.61,
        offset=0.1,
        layer=(102, 0),
        sections=(((103, 0), 0.3, 0.5),),
        radius=None,
        radius_min=None,
        name="native_asymmetric_profile",
    )

    assert isinstance(xs, gf.DAsymmetricCrossSection)
    assert xs.name == "native_asymmetric_profile"
    assert xs.get_sections()[0].section_min == pytest.approx(-0.205, abs=gf.kcl.dbu)


def test_xsection_requires_native_factory() -> None:
    @gf.cross_section.xsection
    def native_profile() -> gf.DCrossSection:
        return gf.cross_section.cross_section(
            width=0.7, layer=(104, 0), radius=None, radius_min=None
        )

    xs = native_profile()
    assert isinstance(xs, gf.DCrossSection)
    assert xs.name == "native_profile"


def test_cross_section_warns_when_dropping_legacy_metadata() -> None:
    with pytest.warns(gf.cross_section.CrossSectionWarning, match="port_names"):
        xs = gf.cross_section.cross_section(
            width=0.7,
            layer=(105, 0),
            port_names=("e1", "e2"),
            radius=None,
            radius_min=None,
            name="native_metadata_adapter",
        )

    assert isinstance(xs, gf.DCrossSection)


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
    def basic_xs(width: float = 1.0) -> gf.DCrossSection:
        return gf.cross_section.cross_section(width=width, layer=(1, 0))

    assert gf.cross_section.is_cross_section("basic_xs", basic_xs)


def test_is_cross_section_subclass() -> None:
    class OtherCrossSection(gf.DCrossSection):
        pass

    def cross_section(**kwargs: Any) -> OtherCrossSection:
        return OtherCrossSection(**kwargs)

    assert gf.cross_section.is_cross_section("cross_section", cross_section)


def test_is_cross_section_subclass_name_not_including_cross_section() -> None:
    class SubclassCrossSection(gf.DCrossSection):
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


def test_section_requires_width_value_or_function() -> None:
    with pytest.raises(ValidationError):
        gf.Section(layer=(1, 0))


@given(
    t_points=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    ),
    cladding_offset=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    w_base=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    w_slope=st.floats(
        min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
)
def test_cross_section_callable_width_offset(
    t_points: npt.NDArray[np.float64],
    cladding_offset: float,
    w_base: float,
    w_slope: float,
) -> None:
    def width_fn(t: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        return w_base + w_slope * t

    def offset_fn(t: float) -> float:
        return 0.1 * t

    nominal_width = float(width_fn(0.5))
    nominal_offset = offset_fn(0.5)
    assume(nominal_width > 0.01)

    with pytest.warns(gf.cross_section.CrossSectionWarning):
        xs = gf.cross_section.cross_section(
            width=width_fn,
            offset=offset_fn,
            layer=(1, 0),
            cladding_layers=((2, 0),),
            cladding_offsets=(cladding_offset,),
        )

    core, cladding = xs.get_sections()
    assert core.width == pytest.approx(nominal_width, abs=2 * gf.kcl.dbu)
    assert core.section_min == pytest.approx(
        nominal_offset - nominal_width / 2, abs=gf.kcl.dbu
    )
    assert cladding.width == pytest.approx(
        nominal_width + 2 * cladding_offset, abs=2 * gf.kcl.dbu
    )


def test_is_cross_section_private() -> None:
    def _private_xs() -> gf.DCrossSection:
        return gf.cross_section.cross_section(width=1.0, layer=(1, 0))

    assert not gf.cross_section.is_cross_section("_private_xs", _private_xs)


def test_taper_cross_section_instance_matches_name() -> None:
    """Taper must honor width overrides for a native cross-section.

    Passing a native cross-section instance must preserve the taper's width
    overrides (and not reuse one geometry for both ports), matching the
    equivalent string spec and avoiding cell-cache poisoning (#4588).
    """
    from gdsfactory.cross_section import xsection

    @xsection
    def _xs_4588(width: float = 0.5) -> gf.DCrossSection:
        return gf.cross_section.cross_section(
            width=width,
            layer=(1, 0),
            port_names=("o1", "o2"),
            port_types=("optical", "optical"),
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
