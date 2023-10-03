import gdsfactory as gf


def test_clear_cache() -> None:
    c1 = gf.c.straight()
    gf.clear_cache()
    c2 = gf.c.straight()
    assert c1.name == c2.name
