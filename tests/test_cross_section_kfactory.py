from __future__ import annotations

from math import isclose

import kfactory as kf

import gdsfactory as gf


def _section_values(
    cross_section: kf.DCrossSection | kf.DAsymmetricCrossSection,
) -> list[tuple[int, int, float, float]]:
    return [
        (
            section.layer.layer,
            section.layer.datatype,
            section.section_min,
            section.section_max,
        )
        for section in cross_section.get_sections()
    ]


def _assert_section_values(
    cross_section: kf.DCrossSection | kf.DAsymmetricCrossSection,
    expected: list[tuple[int, int, float, float]],
) -> None:
    actual = _section_values(cross_section)
    assert len(actual) == len(expected)
    for actual_section, expected_section in zip(actual, expected, strict=True):
        assert actual_section[:2] == expected_section[:2]
        assert isclose(actual_section[2], expected_section[2])
        assert isclose(actual_section[3], expected_section[3])


def test_kfactory_cross_section_is_symmetric() -> None:
    gf.gpdk.PDK.activate()
    xs = gf.cross_section.kfactory_cross_section(
        width=0.5,
        layer="WG",
        cladding_layers=("SLAB90",),
        cladding_offsets=2.0,
        bbox_layers=("DEVREC",),
        bbox_offsets=(1.0,),
        radius=10.0,
        radius_min=7.0,
        name="kfactory_symmetric",
    )

    assert isinstance(xs, kf.DCrossSection)
    assert xs.name == "kfactory_symmetric"
    assert xs.width == 0.5
    assert xs.radius == 10.0
    assert xs.radius_min == 7.0
    _assert_section_values(
        xs,
        [
            (1, 0, -0.25, 0.25),
            (3, 0, -2.25, 2.25),
        ],
    )
    assert {
        (layer.layer, layer.datatype): offset
        for layer, offset in xs.bbox_sections.items()
    } == {(68, 0): 1.0}


def test_kfactory_cross_section_is_asymmetric() -> None:
    gf.gpdk.PDK.activate()
    xs = gf.cross_section.kfactory_cross_section(
        width=0.5,
        offset=0.1,
        layer="WG",
        sections=(("M1", 0.3, 0.5),),
        radius=None,
        radius_min=None,
        name="kfactory_asymmetric",
    )

    assert isinstance(xs, kf.DAsymmetricCrossSection)
    assert xs.name == "kfactory_asymmetric"
    _assert_section_values(
        xs,
        [
            (1, 0, -0.15, 0.35),
            (41, 0, 0.3, 0.5),
        ],
    )


def test_kfactory_cross_section_preserves_nested_same_layer_sections() -> None:
    gf.gpdk.PDK.activate()
    xs = gf.cross_section.kfactory_cross_section(
        width=0.5,
        layer="WG",
        sections=(("WG", -0.75, 0.75),),
        name="kfactory_nested",
    )

    assert isinstance(xs, kf.DCrossSection)
    _assert_section_values(
        xs,
        [
            (1, 0, -0.25, 0.25),
            (1, 0, -0.75, 0.75),
        ],
    )


def test_kfactory_cross_section_snaps_edges_before_classifying_symmetry() -> None:
    gf.gpdk.PDK.activate()
    xs = gf.cross_section.kfactory_cross_section(
        width=0.5,
        offset=0.0005,
        layer="WG",
        name="kfactory_grid_asymmetric",
    )

    assert isinstance(xs, kf.DAsymmetricCrossSection)
    _assert_section_values(xs, [(1, 0, -0.25, 0.251)])
