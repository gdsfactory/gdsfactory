import pytest

import gdsfactory as gf
from gdsfactory.components.texts.text import TextFont


def test_text_custom_font() -> None:
    font = TextFont(
        name="test_blocks",
        glyphs={
            "A": [[(0, 0), (400, 0), (400, 1000), (0, 1000)]],
            "B": [[(0, 0), (600, 0), (600, 1000), (0, 1000)]],
        },
        widths={"A": 400, "B": 600},
        indents={"A": 100, "B": 200},
        space_width=300,
    )

    component = gf.c.text(text="A B", size=10, font=font, layer=(1, 0))
    renamed_font = TextFont(
        name="test_blocks_v2",
        glyphs=font.glyphs,
        widths=font.widths,
        indents=font.indents,
        space_width=font.space_width,
    )
    renamed_component = gf.c.text(text="A B", size=10, font=renamed_font, layer=(1, 0))

    assert component.name.startswith("text_gdsfactorypcomponentsptextsptext_")
    assert component.name != renamed_component.name
    assert component.dxsize == pytest.approx(14)


def test_text_custom_font_requires_metrics() -> None:
    font = TextFont(
        name="test_missing_metrics",
        glyphs={"A": [[(0, 0), (400, 0), (400, 1000), (0, 1000)]]},
        widths={},
        indents={},
    )

    with pytest.raises(ValueError, match="Missing width or indent"):
        gf.c.text(text="A", font=font)
