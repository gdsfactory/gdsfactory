from __future__ import annotations

import pytest

import gdsfactory as gf


def test_text_freetype_renders_geometry() -> None:
    """FreeType glyph extraction must not crash or return empty text."""
    component = gf.components.text_freetype("abc")

    assert component.dbbox().width() > 0
    assert component.dbbox().height() > 0
    assert component.get_polygons()


@pytest.mark.parametrize("justify", ["left", "center", "right"])
def test_text_freetype_justification(justify: str) -> None:
    component = gf.components.text_freetype("qwerty", justify=justify)

    if justify == "left":
        assert component.dxmin == pytest.approx(0)
    elif justify == "center":
        assert component.dx == pytest.approx(0)
    else:
        assert component.dxmax == pytest.approx(0)


def test_text_freetype_rejects_invalid_justification() -> None:
    with pytest.raises(ValueError, match="not in"):
        gf.components.text_freetype("abc", justify="invalid")
