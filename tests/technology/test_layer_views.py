import pytest

from gdsfactory.technology import LayerStack, LayerView, LayerViews
from gdsfactory.technology.layer_map import LayerMap
from gdsfactory.technology.layer_stack import LayerLevel
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
    LAYER_VIEWS = PDK.layer_views
    c = LAYER_VIEWS.preview_layerset()
    assert c


def test_hatch_pattern_custom_pattern() -> None:
    from gdsfactory.technology.layer_views import HatchPattern

    hatch_pattern = HatchPattern(name="test", custom_pattern="**\n**\n")
    assert hatch_pattern.custom_pattern == "**\n**\n"
    hatch_pattern = HatchPattern(name="test", custom_pattern=None)
    assert hatch_pattern.custom_pattern is None

    with pytest.raises(ValueError):
        HatchPattern(name="test", custom_pattern="*" * 33 + "\n")


def test_hatch_pattern_to_klayout_xml() -> None:
    from gdsfactory.technology.layer_views import HatchPattern

    hatch_pattern = HatchPattern(name="test", custom_pattern="**\n**\n")
    assert hatch_pattern.to_klayout_xml()


if __name__ == "__main__":
    test_hatch_pattern_to_klayout_xml()
    test_hatch_pattern_custom_pattern()
