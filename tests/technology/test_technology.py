from gdsfactory.config import PATH
from gdsfactory.technology import LayerViews


def test_yaml() -> None:
    tech_dir = PATH.repo / "extra" / "test_tech"

    # Load from existing layer properties file
    lyp = LayerViews.from_lyp(str(PATH.klayout_lyp))

    # Export layer properties to yaml files
    layer_yaml = str(tech_dir / "layers.yml")
    lyp.to_yaml(layer_yaml)

    # Load layer properties from yaml files and check that they're the same
    lyp_loaded = LayerViews.from_yaml(layer_yaml)
    assert lyp_loaded == lyp
