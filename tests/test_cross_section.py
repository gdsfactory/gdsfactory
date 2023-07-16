from __future__ import annotations

import gdsfactory as gf


def test_waveguide_setting() -> None:
    x = gf.cross_section.cross_section(width=2)
    assert x.width == 2


def test_settings_different() -> None:
    strip1 = gf.cross_section.strip()
    strip2 = gf.cross_section.strip(layer=(2, 0))
    assert strip1.info["settings"]["layer"] == "WG"
    assert strip2.info["settings"]["layer"] == [2, 0]


if __name__ == "__main__":
    # test_waveguide_setting()
    test_settings_different()
