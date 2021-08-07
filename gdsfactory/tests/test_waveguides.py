from gdsfactory.cross_section import get_cross_section


def test_waveguide_setting():
    x = get_cross_section("strip", width=2)
    assert x.info["width"] == 2


def test_waveguide_str():
    x = get_cross_section(dict(component="strip", width=2))
    assert x.info["width"] == 2


def test_waveguide_str_and_setting():
    x = get_cross_section(dict(component="strip", width=2))
    assert x.info["width"] == 2


if __name__ == "__main__":
    a = 5
    x = get_cross_section(waveguide="strip", width=2)
