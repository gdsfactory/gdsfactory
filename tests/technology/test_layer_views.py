import pytest
from pydantic_extra_types.color import Color

from gdsfactory.technology import LayerStack, LayerView, LayerViews
from gdsfactory.technology.layer_map import LayerMap
from gdsfactory.technology.layer_stack import LayerLevel
from gdsfactory.technology.layer_views import HatchPattern, LineStyle
from gdsfactory.typings import Layer

nm = 1e-3


class LayerMapFabA(LayerMap):
    WG: Layer = (34, 0)
    SLAB150: Layer = (2, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TEXT: Layer = (66, 0)


LAYER = LayerMapFabA


class FabALayerViews(LayerViews):
    WG: LayerView = LayerView(color="gold")
    SLAB150: LayerView = LayerView(color="red")
    TE: LayerView = LayerView(color="green")


LAYER_VIEWS = FabALayerViews(layers=LAYER)


def get_layer_stack_faba(
    thickness_wg: float = 500 * nm, thickness_slab: float = 150 * nm
) -> LayerStack:
    """Returns fabA LayerStack."""
    return LayerStack(
        layers=dict(
            strip=LayerLevel(
                layer=LAYER.WG,
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
            ),
            strip2=LayerLevel(
                layer=LAYER.SLAB150,
                thickness=thickness_slab,
                zmin=0.0,
                material="si",
            ),
        )
    )


def test_preview_layerset() -> None:
    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    LAYER_VIEWS = PDK.get_layer_views()
    c = LAYER_VIEWS.preview_layerset()
    assert c


def test_hatch_pattern_custom_pattern() -> None:
    hatch_pattern = HatchPattern(name="test", custom_pattern="**\n**\n")
    assert hatch_pattern.custom_pattern == "**\n**\n"
    hatch_pattern = HatchPattern(name="test", custom_pattern=None)
    assert hatch_pattern.custom_pattern is None

    with pytest.raises(ValueError):
        HatchPattern(name="test", custom_pattern="*" * 33 + "\n")


def test_hatch_pattern_to_klayout_xml() -> None:
    hatch_pattern = HatchPattern(name="test", custom_pattern="**\n**\n")
    res = hatch_pattern.to_klayout_xml()
    assert len(res) > 0

    hatch_pattern = HatchPattern(name="test", custom_pattern=None)
    with pytest.raises(KeyError):
        hatch_pattern.to_klayout_xml()

    hatch_pattern = HatchPattern(name="test", custom_pattern="**")
    res = hatch_pattern.to_klayout_xml()
    assert len(res) > 0


def test_line_style_custom_style() -> None:
    line_style = LineStyle(name="test", custom_style="**")
    assert line_style.custom_style == "**"
    line_style = LineStyle(name="test", custom_style=None)
    assert line_style.custom_style is None

    with pytest.raises(ValueError):
        LineStyle(name="test", custom_style="invalid$chars")

    with pytest.raises(ValueError):
        LineStyle(name="test", custom_style="*" * 33)


def test_line_style_to_klayout_xml() -> None:
    line_style = LineStyle(name="test", custom_style="**")
    res = line_style.to_klayout_xml()
    assert len(res) > 0

    line_style = LineStyle(name="test", custom_style=None)
    with pytest.raises(KeyError):
        line_style.to_klayout_xml()


def test_layer_view_init() -> None:
    lv = LayerView(gds_layer=1, gds_datatype=0)
    assert lv.layer == (1, 0)

    lv = LayerView(color="#FF0000")
    assert lv.fill_color == Color("#FF0000")
    assert lv.frame_color == Color("#FF0000")

    lv = LayerView(brightness=50)
    assert lv.fill_brightness == 50
    assert lv.frame_brightness == 50

    with pytest.raises(KeyError):
        LayerView(layer=(1, 0), gds_layer=1, gds_datatype=0)

    with pytest.raises(KeyError):
        LayerView(color="#FF0000", fill_color="#00FF00")

    with pytest.raises(KeyError):
        LayerView(brightness=50, fill_brightness=60)


def test_layer_view_dict() -> None:
    lv = LayerView(color="#FF0000")

    d = lv.dict(simplify=False)
    assert "fill_color" in d
    assert "frame_color" in d
    assert "color" not in d
    assert d["fill_color"] == d["frame_color"] == Color("#FF0000")

    d = lv.dict(simplify=True)
    assert "fill_color" not in d
    assert "frame_color" not in d
    assert "color" in d
    assert d["color"] == Color("#FF0000")


def test_layer_view_str() -> None:
    lv = LayerView(color="#00FF00")
    assert str(lv)


if __name__ == "__main__":
    lv = LayerView(color="#00FF00")
    print(repr(lv))
