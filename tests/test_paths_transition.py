from __future__ import annotations

from typing import Any

import pytest

import gdsfactory as gf


def test_path_transition_class() -> None:
    P = gf.path.straight(length=10, npoints=101)

    s0 = ((1, 0), -0.5, 0.5)
    s1 = ((3, 0), -1.5, 1.5)
    X1 = gf.cross_section.strip(width=None, sections=(s0, s1))
    X2 = gf.cross_section.strip(width=None, sections=(s0,))

    T = gf.path.transition(X1, X2)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_asymmetric_class() -> None:
    P = gf.path.straight(length=10, npoints=101)

    s0 = ((1, 0), -0.5, 0.5)
    s1 = ((3, 0), -1.5, 1.5)
    X1 = gf.cross_section.strip(width=None, sections=(s0, s1))
    X2 = gf.cross_section.strip(width=None, sections=(s0,))

    T = gf.path.transition_asymmetric(X1, X2)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_function() -> None:
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.strip(width=1)
    X2 = gf.cross_section.strip(width=3)
    T = gf.path.transition(X1, X2)
    P = gf.path.straight(length=10, npoints=101)
    c = gf.path.extrude_transition(P, T)
    assert c


def test_path_transition_asymmetric_function() -> None:
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.strip(width=1)
    X2 = gf.cross_section.strip(width=3)
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
    assert xs1_rt.get_sections()[0].layer == xs1.get_sections()[0].layer

    t2 = gf.components.taper_cross_section(xs1_rt, xs2, length=10, linear=True)
    assert any(t2.get_polygons(merge=False).values())


def _swapped_stacks() -> tuple[gf.CrossSection, gf.CrossSection]:
    """A WG core in SLAB90, then a SLAB90 core in WG: the main strips cross layers."""

    def stack(*names: str) -> gf.CrossSection:
        return gf.cross_section.strip(
            width=None, sections=[(n, -0.5 - i, 0.5 + i) for i, n in enumerate(names)]
        )

    return stack("WG", "SLAB90"), stack("SLAB90", "WG")


def _main_port(c: gf.Component, name: str) -> tuple[float, str]:
    port = c.ports[name]
    return port.width, gf.get_layer_info(port.layer).name


def test_transition_main_ports_ignore_explicit_cross_layer_pairs() -> None:
    a, b = _swapped_stacks()

    c = gf.path.extrude_transition(
        gf.path.straight(10),
        gf.path.transition(a, b),
        section_pairs=[(0, 1), (1, 0)],
    )
    assert set(c.get_polygons()) == {gf.get_layer("WG"), gf.get_layer("SLAB90")}
    assert _main_port(c, "o1") == (1.0, "WG")
    assert _main_port(c, "o2") == (1.0, "SLAB90")


@pytest.mark.parametrize("section_pairs", [None, [(0, 0)], []])
def test_transition_that_draws_nothing_raises(
    section_pairs: list[tuple[int, int]] | None,
) -> None:
    """Main strips on different layers are not paired with auxiliary strips by default."""
    a, b = _swapped_stacks()

    with pytest.raises(ValueError, match="nothing to draw"):
        gf.path.extrude_transition(
            gf.path.straight(10),
            gf.path.transition(a, b),
            section_pairs=section_pairs,
        )


def test_transition_omits_main_ports_when_main_strips_are_unpaired() -> None:
    a, _ = _swapped_stacks()

    c = gf.path.extrude_transition(
        gf.path.straight(10), gf.path.transition(a, a), section_pairs=[(1, 1)]
    )
    assert set(c.get_polygons()) == {gf.get_layer("SLAB90")}
    assert not list(c.ports)


@pytest.mark.parametrize("omit", [{"hidden": {1}}, {"skip_transition": [1]}])
def test_transition_omitting_its_only_drawable_strip_raises(
    omit: dict[str, Any],
) -> None:
    """A WG core in SLAB90 to a SLAB90 core: only the slab can be drawn."""
    a, _ = _swapped_stacks()
    slab = gf.cross_section.strip(width=None, sections=[("SLAB90", -0.5, 0.5)])

    with pytest.raises(ValueError, match="nothing to draw"):
        gf.path.extrude_transition(
            gf.path.straight(10),
            gf.path.transition(a, slab),
            section_pairs=[(1, 0)],
            **omit,
        )


def test_transition_rib_to_slab_needs_an_explicit_pair() -> None:
    rib = gf.cross_section.rib2(width=0.5, width_slab=5.5)
    slab = gf.cross_section.strip(width=5.5, layer="SLAB90")
    transition = gf.path.transition(rib, slab)

    with pytest.raises(ValueError, match="nothing to draw"):
        gf.path.extrude_transition(gf.path.straight(10), transition)

    c = gf.path.extrude_transition(
        gf.path.straight(10), transition, section_pairs=[(1, 0)]
    )
    assert set(c.get_polygons()) == {gf.get_layer("SLAB90")}
    assert _main_port(c, "o1") == (0.5, "WG")
    assert _main_port(c, "o2") == (5.5, "SLAB90")
