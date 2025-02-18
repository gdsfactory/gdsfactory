import pytest

import gdsfactory as gf


def test_align() -> None:
    c = gf.components.align_wafer(layer_cladding=(3, 0))
    assert not c.bbox(gf.get_layer((3, 0))).empty()

    c = gf.components.align_wafer()
    assert c.bbox(gf.get_layer((3, 0))).empty()


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
