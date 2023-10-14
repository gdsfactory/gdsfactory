import pytest

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk


@pytest.fixture
def pdk() -> gf.Pdk:
    """Returns the generic PDK"""
    return get_generic_pdk()


def test_get_cross_section() -> None:
    assert gf.pdk.get_cross_section("xs_sc") == gf.cross_section.xs_sc
    assert gf.pdk.get_cross_section(gf.cross_section.strip()) == gf.cross_section.xs_sc
    cross_section = {"cross_section": "xs_sc", "settings": {"width": 1}}
    xs = gf.get_cross_section(cross_section)
    assert xs.sections[0].width == 1


def test_klayout_technology(pdk) -> None:
    tech = pdk.klayout_technology
    assert tech.name == pdk.name
    assert tech.layer_views == pdk.layer_views
    assert tech.connectivity == pdk.connectivity
    assert tech.layer_map == pdk.layers
    assert tech.layer_stack == pdk.layer_stack


def test_klayout_technology_write(pdk, tmp_path) -> None:
    tech = pdk.klayout_technology
    tech.write_tech(tech_dir=tmp_path)
    assert (tmp_path / "layers.lyp").exists()
    assert (tmp_path / "tech.lyt").exists()
