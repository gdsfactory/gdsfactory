import gdsfactory as gf


def test_partial_cross_section() -> None:
    strip200 = gf.partial(gf.cross_section.strip, width=0.2)
    strip400 = gf.partial(gf.cross_section.strip, width=0.4)
    c200 = gf.components.straight(cross_section=strip200)
    c400 = gf.components.straight(cross_section=strip400)
    assert c200.name != c400.name, f"{c200.name} {c400.name}"


if __name__ == "__main__":
    test_partial_cross_section()
    # f = gf.partial(gf.cross_section.strip, width=0.2)
