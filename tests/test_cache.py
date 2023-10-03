import gdsfactory as gf
from gdsfactory.add_padding import add_padding_to_size_container


def test_cache_container() -> None:
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid  # pulling this from cache


def test_cache_name() -> None:
    c = gf.components.straight()
    c1 = add_padding_to_size_container(c, xsize=100, ysize=100)
    c2 = add_padding_to_size_container(c, xsize=100, ysize=100)
    assert c1.name == c2.name
