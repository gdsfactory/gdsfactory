"""Extrusion settings stored on cross sections.

Each test uses its own main width, so that its geometries (and their extrude
configs, which are keyed by geometry) don't clash with other tests' in the
shared layout.
"""

from __future__ import annotations

import kfactory as kf
import pytest
from kfactory.exceptions import ExtrudeConfigConflictError

import gdsfactory as gf
from gdsfactory.cross_section import Section, cross_section


def _port_names(c: gf.Component) -> dict[str, str]:
    return {p.name: p.port_type for p in c.ports if p.name is not None}


def test_no_settings_writes_no_config(kcl: kf.KCLayout) -> None:
    xs = cross_section(width=0.5, sections=[("SLAB90", -3, 3)], kcl=kcl)
    assert kcl.get_extrude_config(xs) is None
    assert kcl.extrude_configs == {}


def test_config_written_once(kcl: kf.KCLayout) -> None:
    xs = cross_section(width=0.5, port_names=("in", "out"), kcl=kcl)
    config = kcl.get_extrude_config(xs)
    assert config is not None
    assert config.ports == {0: ("in", "out", "optical")}
    cross_section(width=0.5, port_names=("in", "out"), kcl=kcl)
    assert kcl.get_extrude_config(xs) is config


def test_conflicting_config_raises(kcl: kf.KCLayout) -> None:
    cross_section(width=0.5, port_names=("in", "out"), kcl=kcl)
    with pytest.raises(ExtrudeConfigConflictError):
        cross_section(width=0.5, port_names=("a", "b"), kcl=kcl)


def test_named_and_unnamed_share_config(kcl: kf.KCLayout) -> None:
    xs = cross_section(width=0.5, port_names=("in", "out"), kcl=kcl)
    named = cross_section(width=0.5, name="shared_config_xs", kcl=kcl)
    assert kcl.get_extrude_config(named) is kcl.get_extrude_config(xs)


def test_section_settings_map_to_normalized_index(kcl: kf.KCLayout) -> None:
    # get_sections() sorts strips, so the heater ends up before the slab.
    xs = cross_section(
        width=0.5,
        sections=[
            ("SLAB90", -3, 3),
            Section("HEATER", -1.25, 1.25, insets=(5, 5), hidden=True, simplify=0.01),
        ],
        kcl=kcl,
    )
    config = kcl.get_extrude_config(xs)
    assert config is not None
    (index,) = (
        i
        for i, s in enumerate(xs.get_sections())
        if (s.layer.layer, s.layer.datatype) == (47, 0)
    )
    assert config.insets == {index: (5000, 5000)}
    assert config.hidden == {index}
    assert config.simplify == {index: 10}
    assert config.ports is None


def test_cladding_simplify(kcl: kf.KCLayout) -> None:
    xs = cross_section(
        width=0.5,
        cladding_layers=("SLAB90", "DEEP_ETCH"),
        cladding_offsets=(3, 1),
        cladding_simplify=(0.05, None),
        kcl=kcl,
    )
    config = kcl.get_extrude_config(xs)
    assert config is not None
    (index,) = (
        i
        for i, s in enumerate(xs.get_sections())
        if (s.layer.layer, s.layer.datatype) == (3, 0)
    )
    assert config.simplify == {index: 50}


def test_merged_section_with_settings_raises(kcl: kf.KCLayout) -> None:
    with pytest.raises(ValueError, match="merged"):
        cross_section(
            width=0.5,
            sections=[("SLAB90", -3, 1), Section("SLAB90", 0, 3, hidden=True)],
            kcl=kcl,
        )


def test_main_settings_from_first_section(kcl: kf.KCLayout) -> None:
    xs = cross_section(
        width=None,
        sections=[Section("WG", -0.25, 0.25, port_names=("in", "out"))],
        kcl=kcl,
    )
    config = kcl.get_extrude_config(xs)
    assert config is not None
    assert config.ports == {0: ("in", "out", "optical")}


def test_main_settings_twice_raises(kcl: kf.KCLayout) -> None:
    with pytest.raises(ValueError, match="not both"):
        cross_section(
            width=None,
            sections=[Section("WG", -0.25, 0.25, port_names=("in", "out"))],
            port_names=("a", "b"),
            kcl=kcl,
        )


def test_port_types_without_default_ports_raises(kcl: kf.KCLayout) -> None:
    with pytest.raises(ValueError, match="port_names"):
        cross_section(
            width=0.5,
            sections=[Section("SLAB90", -3, 3, port_types="electrical")],
            kcl=kcl,
        )


def test_port_types_must_match(kcl: kf.KCLayout) -> None:
    with pytest.raises(ValueError, match="same type"):
        cross_section(width=0.5, port_types=("optical", "electrical"), kcl=kcl)


def test_extrude_uses_config() -> None:
    xs = cross_section(
        width=0.502,
        port_names=("in", "out"),
        sections=[
            Section(
                "HEATER",
                -1.25,
                1.25,
                port_names=("h1", "h2"),
                port_types="electrical",
                insets=(5, 5),
            )
        ],
    )
    c = gf.path.straight(length=20).extrude(xs)
    assert _port_names(c) == {
        "in": "optical",
        "out": "optical",
        "h1": "electrical",
        "h2": "electrical",
    }
    assert c.ports["h1"].dcenter[0] == pytest.approx(5)
    assert c.ports["h2"].dcenter[0] == pytest.approx(15)
    heater = c.kcl.layer(47, 0)
    assert c.dbbox(heater).width() == pytest.approx(10)


def test_extrude_arguments_override_config() -> None:
    xs = cross_section(
        width=0.504,
        sections=[Section("HEATER", -1.25, 1.25, insets=(5, 5), hidden=True)],
    )
    c = gf.path.straight(length=20).extrude(
        xs, ports={0: ("a", "b", "optical")}, insets={}, hidden=()
    )
    assert _port_names(c) == {"a": "optical", "b": "optical"}
    heater = c.kcl.layer(47, 0)
    assert c.dbbox(heater).width() == pytest.approx(20)


def test_plain_cross_section_inherits_config() -> None:
    with_ports = cross_section(width=0.506, port_names=("in", "out"))
    plain = gf.get_cross_section(
        kf.DCrossSection(kcl=with_ports.kcl, base=with_ports.base)
    )
    c = gf.path.straight(length=10).extrude(plain)
    assert set(_port_names(c)) == {"in", "out"}


def test_with_width_keeps_config() -> None:
    xs = cross_section(width=0.508, port_names=("in", "out"))
    c = gf.path.straight(length=10).extrude(xs, width=0.614)
    assert set(_port_names(c)) == {"in", "out"}
    assert c.ports["in"].dwidth == pytest.approx(0.614)


def test_strip_heater_metal_insets() -> None:
    xs = gf.cross_section.strip_heater_metal(width=0.51, insets=(2, 3))
    c = gf.path.straight(length=20).extrude(xs)
    assert c.ports["e1"].dcenter[0] == pytest.approx(2)
    assert c.ports["e2"].dcenter[0] == pytest.approx(17)
    assert c.ports["o1"].dcenter[0] == pytest.approx(0)


def test_pn_metal_ports() -> None:
    xs = gf.cross_section.pn(width=0.512, layer_via="VIAC", layer_metal="M1")
    c = gf.path.straight(length=10).extrude(xs)
    assert {"e1_top", "e2_top", "e1_bot", "e2_bot"} <= set(_port_names(c))
    assert c.ports["e1_top"].port_type == "electrical"
