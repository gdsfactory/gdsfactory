from gdsfactory.config import PATH
from gdsfactory.technology import LayerViews


def test_yaml() -> None:
    LayerViews.from_lyp(str(PATH.klayout_lyp))

    # tech_dir = PATH.repo / "extra" / "test_tech"
    # layer_yaml = str(tech_dir / "layers.yml")
    # lyp.to_yaml(layer_yaml)

    # lyp_loaded = LayerViews.from_yaml(layer_yaml)
    # assert lyp_loaded == lyp
