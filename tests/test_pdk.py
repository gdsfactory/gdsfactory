import pytest

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_get_cross_section() -> None:
    assert gf.pdk.get_cross_section("strip") == gf.cross_section.strip()
    assert (
        gf.pdk.get_cross_section(gf.cross_section.strip()) == gf.cross_section.strip()
    )
    cross_section = {"cross_section": "strip", "settings": {"width": 1}}
    xs = gf.get_cross_section(cross_section)
    assert xs.sections[0].width == 1


def test_get_layer() -> None:
    assert gf.get_layer(1) == LAYER.WG
    assert gf.get_layer((1, 0)) == LAYER.WG
    assert gf.get_layer("WG") == LAYER.WG


def test_container_cell_conflict_raises_error() -> None:
    """Test that a cell with the same name as a container raises an error."""
    pdk = gf.Pdk(
        name="test",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        cells={
            "straight": gf.components.straight,
            "add_pads_top": gf.containers.add_pads_top,
        },
        containers={"add_pads_top": gf.containers.add_pads_top},
    )

    with pytest.raises(ValueError, match=".* overlapping cell names .*add_pads_top.*"):
        pdk.get_component("add_pads_top")
