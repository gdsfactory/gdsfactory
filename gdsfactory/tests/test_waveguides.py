import gdsfactory as gf


def test_waveguide_setting():
    x = gf.cross_section.cross_section(width=2)
    assert x.info["width"] == 2


if __name__ == "__main__":
    test_waveguide_setting()
