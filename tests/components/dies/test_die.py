import pytest

import gdsfactory as gf


@pytest.mark.parametrize("draw_corners", [True, False])
def test_die(draw_corners: bool) -> None:
    c = gf.components.die(layer=(4, 0), bbox_layer=(5, 0), draw_corners=draw_corners)
    assert not c.bbox(gf.get_layer((4, 0))).empty()
    assert not c.bbox(gf.get_layer((5, 0))).empty()


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
