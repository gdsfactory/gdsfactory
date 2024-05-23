import gdsfactory as gf


def test_get_cross_section() -> None:
    assert gf.pdk.get_cross_section("strip") == gf.cross_section.strip
    assert gf.pdk.get_cross_section(gf.cross_section.strip()) == gf.cross_section.strip
    cross_section = {"cross_section": "strip", "settings": {"width": 1}}
    xs = gf.get_cross_section(cross_section)
    assert xs.sections[0].width == 1
