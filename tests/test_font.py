from __future__ import annotations

import gdsfactory as gf


def test_text_freetype_renders_geometry() -> None:
    """FreeType glyph extraction must not crash or return empty text."""
    component = gf.components.text_freetype("abc")

    assert component.dbbox().width() > 0
    assert component.dbbox().height() > 0
    assert component.get_polygons()
