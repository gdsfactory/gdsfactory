from __future__ import annotations

from functools import partial
from typing import Any

import jsondiff
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
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
    assert x.width == width


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


def test_transition_asymmetric_names() -> None:
    layer = (1, 0)
    s1 = gf.Section(width=5, layer=layer, port_names=("o1", "o2"), name="core")
    s2 = gf.Section(width=50, layer=layer, port_names=("o1", "o2"), name="core")

    xs1 = gf.CrossSection(sections=(s1,))
    xs2 = gf.CrossSection(sections=(s2,))
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


def _width_function(t: float) -> float:
    return 0.5 + 2 * t


def _offset_function(t: float) -> float:
    return 0.0


def _other_offset_function(t: float) -> float:
    return 2.0 * t


def _profile_cross_section() -> gf.CrossSection:
    return gf.cross_section.cross_section(
        width=_width_function, offset=_offset_function
    )


def test_copy_keeps_falsy_overrides() -> None:
    """A nominal width of 0 and the layer with index 0 are overrides, not omissions."""
    xs = gf.get_cross_section("strip")

    # width_function drives the extrusion, so width is only a nominal value
    taper = xs.copy(width=0.0, width_function=_width_function)
    assert taper.width == 0.0
    assert taper.sections[0].width_function is _width_function
    c = gf.path.extrude(gf.path.straight(length=10), cross_section=taper)
    assert np.isclose(c.ysize, _width_function(1.0))

    # first entry of a layer enum has index 0, which is falsy
    assert xs.copy(layer=LAYER.WAFER).layer == LAYER.WAFER

    # an offset back to the center is an override, not an omission
    assert (
        gf.cross_section.cross_section(offset=1.0).copy(offset=0.0).sections[0].offset
        == 0.0
    )


def test_copy_rejects_none_for_fields_that_take_no_none() -> None:
    """Omitting an override is the default, so None is left to mean what the field says."""
    xs = gf.get_cross_section("strip")

    overrides: list[dict[str, Any]] = [
        {"width": None},
        {"offset": None},
        {"layer": None},
    ]
    for override in overrides:
        with pytest.raises(ValidationError):
            xs.copy(**override)


def test_copy_keeps_profile_functions() -> None:
    """A copy that does not pass the profile functions has to keep them."""
    xs = _profile_cross_section()

    # an override that leaves the profile alone keeps it silently
    unrelated = xs.copy(radius=20)
    assert unrelated.sections[0].width_function is _width_function
    assert unrelated.sections[0].offset_function is _offset_function

    # a width override cannot win against the width_function it inherits, so it warns
    with pytest.warns(UserWarning, match="only nominal"):
        nominal_wider = xs.copy(width=3.0)
    assert nominal_wider.sections[0].width_function is _width_function
    assert nominal_wider.sections[0].offset_function is _offset_function
    assert nominal_wider.width == 3.0

    # width_function survived, so it still drives the extrusion over the new width
    c = gf.path.extrude(gf.path.straight(length=10), cross_section=nominal_wider)
    assert np.isclose(c.ysize, _width_function(1.0))


def test_copy_removes_profile_function_given_none() -> None:
    """None is an override that removes the function, unlike omitting it."""
    xs = _profile_cross_section()

    # the width now has nothing competing with it, so no warning and no taper
    literal = xs.copy(width=1.0, width_function=None)
    assert literal.sections[0].width_function is None
    assert literal.sections[0].offset_function is _offset_function
    c = gf.path.extrude(gf.path.straight(length=10), cross_section=literal)
    assert np.isclose(c.ysize, 1.0)

    # removing one profile function leaves the other one alone
    assert xs.copy(offset_function=None).sections[0].width_function is _width_function
    assert xs.copy(offset_function=None).sections[0].offset_function is None

    # a callable width leaves a nominal width of 0 behind, so the section needs one
    with pytest.raises(ValidationError):
        xs.copy(width_function=None)


def test_copy_overrides_offset() -> None:
    """`offset` is an override of its own, next to the function that can outrank it."""
    assert gf.get_cross_section("strip").copy(offset=1.0).sections[0].offset == 1.0

    xs = _profile_cross_section()

    # the offset_function drives the extrusion, so the offset is only nominal, and warns
    with pytest.warns(UserWarning, match="only nominal"):
        nominal = xs.copy(offset=1.0)
    assert nominal.sections[0].offset == 1.0
    assert nominal.sections[0].offset_function is _offset_function

    # removing the function leaves the offset alone to drive it, so nothing warns
    literal = xs.copy(offset=1.0, offset_function=None)
    assert literal.sections[0].offset == 1.0
    assert literal.sections[0].offset_function is None


def test_copy_replaces_only_the_profile_function_it_is_given() -> None:
    """Passing one profile function must not clear the other one."""
    xs = _profile_cross_section()

    other_width_function = gf.path.transition_exponential(1.0, 5.0)
    new_width = xs.copy(width_function=other_width_function).sections[0]
    assert new_width.width_function is other_width_function
    assert new_width.offset_function is _offset_function

    new_offset = xs.copy(offset_function=_other_offset_function).sections[0]
    assert new_offset.width_function is _width_function
    assert new_offset.offset_function is _other_offset_function


def test_copy_keeps_replacement_sections_verbatim() -> None:
    """Replacement sections used to get the original width and layer written over them."""
    xs = _profile_cross_section()
    other = gf.Section(width=4.0, layer="SLAB90", name="other")

    assert xs.copy(sections=(other,)).sections == (other,)

    # an explicit empty tuple is an override, not an omission
    assert xs.copy(sections=()).sections == ()


def test_copy_validates_overrides() -> None:
    """Overrides go through Section validation instead of around it."""
    xs = gf.get_cross_section("strip")

    # width of 0 needs a width_function to give the section a width
    with pytest.raises(ValidationError):
        xs.copy(width=0.0)

    with pytest.raises(ValidationError):
        xs.copy(width=-1.0)


def test_copy_validates_replacement_sections() -> None:
    """The whole update is validated, not just the overrides applied to it."""
    xs = gf.get_cross_section("strip")

    built = xs.copy(sections=({"width": 4.0, "layer": "SLAB90"},)).sections[0]
    assert isinstance(built, gf.Section)
    assert built.width == 4.0

    # a mapping has to take the overrides meant for the first section too
    overridden = xs.copy(
        width=2.0, sections=({"width": 4.0, "layer": "SLAB90"},)
    ).sections[0]
    assert overridden.width == 2.0
    assert overridden.layer == "SLAB90"

    with pytest.raises(ValidationError):
        xs.copy(sections=({"width": -1.0, "layer": "SLAB90"},))

    # kwargs are validated too, instead of being written in as given
    with pytest.raises(ValidationError):
        xs.copy(radius="wide")


def test_copy_without_sections_rejects_overrides() -> None:
    """There is no first section to apply the overrides to."""
    with pytest.raises(ValueError, match="no section to apply"):
        gf.CrossSection().copy(width=1.0)


def test_name() -> None:
    s = gf.cross_section.strip()
    assert s.name == "strip"


def _unnamed(width: float, **kwargs: Any) -> gf.CrossSection:
    """A cross_section whose section name is derived rather than given."""
    return gf.CrossSection(
        sections=(gf.Section(width=width, layer="WG", **kwargs),), radius=10
    )


def test_derived_name_does_not_depend_on_provenance() -> None:
    """The same cross_section has the same name however it was built.

    Cross_section names reach cell names, Component.info and netlists, so a name
    that depends on the code path churns golden data.
    """
    fresh = _unnamed(2.0)
    copied = _unnamed(1.0).copy(width=2.0)

    assert copied.name == fresh.name, f"{copied.name} != {fresh.name}"
    assert copied.sections[0].name == fresh.sections[0].name

    # the name is a private attribute, which pydantic compares, so a name that
    # depends on the code path makes equal cross_sections compare unequal too
    assert copied == fresh


def test_derived_name_is_a_prefix_of_the_hash() -> None:
    """One derivation, so the short name cannot drift from the full digest."""
    xs = _unnamed(2.0)

    assert len(xs.hash) == 32
    assert xs.name == f"xs_{xs.hash[:8]}"


def test_derived_name_tracks_content_through_mirror_and_append() -> None:
    """A transformed cross_section must not keep the name of the old content."""
    asymmetric = gf.CrossSection(
        sections=(
            gf.Section(width=0.5, layer="WG", name="core"),
            gf.Section(width=0.2, layer="SLAB90", offset=1.0, name="rib"),
        ),
        radius=10,
    )
    derived = asymmetric.copy(width=1.0)

    def rebuild(xs: gf.CrossSection) -> gf.CrossSection:
        return gf.CrossSection(
            sections=xs.sections, radius=xs.radius, radius_min=xs.radius_min
        )

    mirrored = derived.mirror()
    assert mirrored.name == rebuild(mirrored).name, f"{mirrored.name} is stale"
    assert mirrored.name != derived.name

    appended = derived.append_sections(
        (gf.Section(width=0.3, layer="M1", name="metal"),)
    )
    assert appended.name == rebuild(appended).name, f"{appended.name} is stale"
    assert appended.name != derived.name

    # a transformation that leaves the content alone keeps the given name
    assert gf.get_cross_section("strip").mirror().name == "strip"


def test_copy_re_derives_a_derived_section_name() -> None:
    """A derived section name describes content, so an edit has to derive it again."""
    xs = _unnamed(0.5)
    wide = xs.copy(width=3.0).sections[0]

    assert wide.name != xs.sections[0].name
    # the name a section built with the new content from the start would get
    assert wide.name == _unnamed(3.0).sections[0].name

    # a given name belongs to the caller, so the copy keeps it
    named = gf.CrossSection(sections=(gf.Section(width=0.5, layer="WG", name="core"),))
    assert named.copy(width=3.0).sections[0].name == "core"


def test_mirror_re_derives_section_names() -> None:
    """A mirrored section holds new content, so it gets the name that content gets."""
    off_center = _unnamed(0.5, offset=1.0)
    assert (
        off_center.mirror().sections[0].name
        == _unnamed(0.5, offset=-1.0).sections[0].name
    )
    assert off_center.mirror().mirror() == off_center


def test_derived_names_do_not_depend_on_the_function_object() -> None:
    """Two profile functions that do the same thing have to give the same name.

    A name taken from a function's repr holds its id(), so it moves between
    processes, and cell names, Component.info and netlists move with it.
    """

    def equivalent_width_function(t: float) -> float:
        return 0.5 + 2 * t

    one = gf.Section(width=1.0, layer="WG", width_function=_width_function)
    two = gf.Section(width=1.0, layer="WG", width_function=equivalent_width_function)

    assert one.width_function is not two.width_function
    assert one.name == two.name, f"{one.name} != {two.name}"
    assert (
        gf.CrossSection(sections=(one,)).name == gf.CrossSection(sections=(two,)).name
    )


def test_derived_name_covers_components_along_path() -> None:
    """Pydantic serializes an arbitrary type as {}, which would hide the component."""
    sections = (gf.Section(width=1.0, layer="WG"),)

    def along(component: gf.Component) -> gf.CrossSection:
        return gf.CrossSection(
            sections=sections,
            components_along_path=(
                gf.cross_section.ComponentAlongPath(component=component, spacing=5),
            ),
        )

    rectangle = along(gf.components.rectangle(size=(1, 1)))
    circle = along(gf.components.circle(radius=3))

    assert rectangle != circle
    assert rectangle.name != circle.name, f"{rectangle.name} collides"


def test_hash_stays_out_of_the_fields() -> None:
    """`hash` is derived from the fields, so it must not become one of them.

    A cache in __dict__ would: pydantic iterates __dict__, so the hash would reach
    dict(), the round-trip through it, and every model_copy.
    """
    xs = _unnamed(0.5)
    assert xs.hash == xs.hash

    read = _unnamed(0.5)
    assert read.name  # anything that reads the name would have filled a cache

    assert set(dict(read)) == set(type(read).model_fields)
    assert gf.CrossSection(**dict(read)) == read
    assert read.model_dump() == _unnamed(0.5).model_dump()
    assert read.copy(width=4.0) == _unnamed(4.0)


def test_name_tracks_content_through_model_copy() -> None:
    """A name read before the copy must not follow the old content into it."""
    xs = _unnamed(0.5)
    assert xs.name  # a cached hash would be filled here and copied below

    wider = xs.model_copy(update={"sections": _unnamed(3.0).sections})

    assert wider.name == _unnamed(3.0).name, f"{wider.name} is stale"


def test_derived_name_does_not_depend_on_the_layer_spelling() -> None:
    """A layer named twice over is one layer, so it gives one name."""
    by_name = gf.Section(width=1.0, layer="WG")
    by_enum = gf.Section(width=1.0, layer=LAYER.WG)

    assert by_name.layer != by_enum.layer  # the spelling is kept as given
    assert by_name.name == by_enum.name, f"{by_name.name} != {by_enum.name}"


def test_transition_pairs_derived_sections_with_their_copy() -> None:
    """A derived name moves with every edit, so it cannot be the pairing key.

    Pairing on it left a cross_section with nothing in common with its own copy.
    """
    xs = _unnamed(0.5)

    gf.path.extrude_transition(
        gf.path.straight(10), gf.path.transition(xs, xs.copy(width=2.0))
    )
    gf.components.taper_cross_section(
        cross_section1=xs, cross_section2=xs.copy(width=2.0)
    )


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

    xs = gf.cross_section.cross_section(
        width=width_fn,
        offset=offset_fn,
        layer=(1, 0),
        cladding_layers=((2, 0),),
        cladding_offsets=(cladding_offset,),
    )
    core, cladding = xs.sections

    assert core.width_function is width_fn
    assert core.offset_function is offset_fn
    assert cladding.width_function is not None
    sampled = cladding.width_function(t_points)
    np.testing.assert_allclose(
        sampled,
        width_fn(t_points) + 2 * cladding_offset,
        err_msg="Sampled cladding width does not match expected mathematical output.",
    )


def test_is_cross_section_private() -> None:
    def _private_xs() -> gf.CrossSection:
        return gf.cross_section.cross_section(width=1.0, layer=(1, 0))

    assert not gf.cross_section.is_cross_section("_private_xs", _private_xs)


def test_section_default_name_does_not_mutate_input() -> None:
    from gdsfactory.cross_section import CrossSection, Section

    section = {"width": 2.0, "layer": "SLAB90"}
    expected = dict(section)

    assert Section.model_validate(section).name
    assert section == expected

    assert CrossSection(sections=[section]).sections[0].name
    assert section == expected

    # the derived name is unchanged by the copy: keywords and spec agree
    assert Section(**expected).name == Section.model_validate(expected).name


def test_section_default_name_tracks_edits() -> None:
    from gdsfactory.cross_section import CrossSection

    section = {"width": 2.0, "layer": "SLAB90"}
    narrow = CrossSection(sections=[section]).sections[0]

    section["width"] = 5.0
    wide = CrossSection(sections=[section]).sections[0]

    assert wide.width == 5.0
    assert wide.name != narrow.name

    # two different sections must not collide on an inherited name,
    # which extrude_transition rejects
    xs = gf.CrossSection(sections=(narrow, wide), radius=10)
    assert len({s.name for s in xs.sections}) == 2
    gf.path.extrude_transition(gf.path.straight(10), gf.path.transition(xs, xs))


def test_taper_cross_section_instance_matches_name() -> None:
    """Taper must honor width overrides when cross_section is a CrossSection.

    Passing a CrossSection instance used to drop the taper's width overrides
    (and reuse one geometry for both ports), giving a different result than the
    equivalent string spec and poisoning the cell cache (#4588).
    """
    from gdsfactory.cross_section import CrossSection, Section, xsection

    @xsection
    def _xs_4588(width: float = 0.5) -> CrossSection:
        return CrossSection(
            sections=(
                Section(
                    width=width,
                    layer=(1, 0),
                    port_names=("o1", "o2"),
                    port_types=("optical", "optical"),
                ),
            ),
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
