from __future__ import annotations

import kfactory as kf
import pytest

import gdsfactory as gf


def _symmetric(
    width: float = 0.5,
    name: str | None = None,
    layer: kf.kdb.LayerInfo | None = None,
) -> gf.CrossSection:
    return gf.cross_section.kfactory_cross_section(
        width=width,
        layer=layer or kf.kdb.LayerInfo(1, 0),
        name=name,
    )


def _asymmetric(name: str = "asymmetric") -> gf.CrossSection:
    return gf.cross_section.kfactory_cross_section(
        width=0.5,
        offset=0.1,
        layer=kf.kdb.LayerInfo(1, 0),
        name=name,
    )


def test_cross_section_public_types() -> None:
    symmetric = _symmetric(name="symmetric")
    asymmetric = _asymmetric(name="asymmetric_types")

    assert isinstance(symmetric, gf.SymmetricCrossSection)
    assert isinstance(asymmetric, gf.AsymmetricCrossSection)
    assert isinstance(symmetric, gf.CrossSection)
    assert isinstance(asymmetric, gf.CrossSection)


def test_get_cross_section_returns_d_wrappers() -> None:
    symmetric = _symmetric(name="symmetric")
    asymmetric = _asymmetric(name="asymmetric_types")
    pdk = gf.Pdk(
        name="native-cross-sections",
        cross_sections={
            "symmetric": lambda: symmetric,
            "asymmetric": lambda: asymmetric,
        },
    )

    assert isinstance(pdk.get_cross_section("symmetric"), gf.SymmetricCrossSection)
    assert isinstance(pdk.get_cross_section("asymmetric"), gf.AsymmetricCrossSection)
    assert isinstance(
        pdk.get_cross_section(symmetric.to_itype()), gf.SymmetricCrossSection
    )
    assert isinstance(
        pdk.get_cross_section(asymmetric.to_itype()), gf.AsymmetricCrossSection
    )


def test_get_cross_section_native_instance_does_not_accept_overrides() -> None:
    pdk = gf.Pdk(name="native-cross-sections")
    symmetric = _symmetric(name="symmetric")

    with pytest.raises(TypeError, match="named cross-section factory"):
        pdk.get_cross_section(symmetric, width=2)


def test_register_cross_section_rejects_legacy_factory() -> None:
    pdk = gf.Pdk(name="native-cross-sections")

    def legacy() -> gf.LegacyCrossSection:
        return gf.LegacyCrossSection(sections=(gf.Section(width=0.5, layer=(1, 0)),))

    with pytest.raises(ValueError, match="native CrossSection"):
        pdk.register_cross_sections(legacy=legacy)


def test_pdk_xsection_requires_canonical_default_factory_name() -> None:
    pdk = gf.Pdk(name="native-cross-sections")

    with pytest.raises(ValueError, match="must return a profile named"):

        @pdk.xsection
        def strip(width: float = 0.5) -> gf.CrossSection:
            return _symmetric(width=width)

    pdk = gf.Pdk(name="native-cross-sections-canonical")

    @pdk.xsection
    def canonical_strip(width: float = 0.5) -> gf.CrossSection:
        return _symmetric(
            width=width,
            name="canonical_strip" if width == 0.5 else None,
            layer=kf.kdb.LayerInfo(202, 0),
        )

    xs = pdk.get_cross_section("canonical_strip")
    assert xs.name == "canonical_strip"
    assert xs.kcl.get_base_cross_section("canonical_strip") == xs.base
    assert pdk.get_cross_section("canonical_strip", width=0.7).name != "canonical_strip"


def test_get_cross_section_accepts_raw_kfactory_profiles() -> None:
    pdk = gf.Pdk(name="native-cross-sections")
    symmetric = _symmetric(name="symmetric")
    raw = symmetric.to_itype()

    result = pdk.get_cross_section(raw)

    assert isinstance(result, gf.SymmetricCrossSection)
    assert result.width == symmetric.width


def test_get_cross_section_accepts_raw_asymmetric_um_profile() -> None:
    pdk = gf.Pdk(name="native-cross-sections")
    raw = kf.DAsymmetricalCrossSection(
        layer=kf.kdb.LayerInfo(1, 0),
        section_min=-0.15,
        section_max=0.35,
    )

    result = pdk.get_cross_section(raw)

    assert isinstance(result, gf.AsymmetricCrossSection)
    assert result.section_min == pytest.approx(raw.section_min)
    assert result.section_max == pytest.approx(raw.section_max)
