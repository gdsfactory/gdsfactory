import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.technology import LayerViews


def test_yaml() -> None:
    LayerViews.from_lyp(str(PATH.klayout_lyp))

    # tech_dir = PATH.repo / "extra" / "test_tech"
    # layer_yaml = str(tech_dir / "layers.yml")
    # lyp.to_yaml(layer_yaml)

    # lyp_loaded = LayerViews.from_yaml(layer_yaml)
    # assert lyp_loaded == lyp


def test_preview_layerset() -> None:
    PDK = gf.get_active_pdk()
    LAYER_VIEWS = PDK.layer_views
    c = LAYER_VIEWS.preview_layerset()
    assert c is not None


if __name__ == "__main__":
    PDK = gf.get_active_pdk()
    LAYER_VIEWS = PDK.layer_views
    c = LAYER_VIEWS.preview_layerset()
    c.show()
