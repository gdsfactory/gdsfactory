import pathlib

import gdsfactory as gf
from gdsfactory.technology.layer_views import LayerViews


class LayerMap(gf.LayerEnum):
    """You will need to create a new LayerMap with your specific foundry layers."""

    layout = gf.constant(gf.kcl.layout)


def lyp_to_dataclass(lyp_filepath: str | pathlib.Path, overwrite: bool = True) -> str:
    """Returns python LayerMap script from a klayout layer properties file lyp."""
    filepathin = pathlib.Path(lyp_filepath)
    filepathout = filepathin.with_suffix(".py")

    if filepathout.exists() and not overwrite:
        raise FileExistsError(f"You can delete {filepathout}")

    script = """
from gdsfactory.typings import Layer
from gdsfactory.technology.layer_map import LayerMap


class LayerMapFab(LayerMap):
"""
    lys = LayerViews.from_lyp(filepathin)
    for layer_name, layer in sorted(lys.get_layer_views().items()):
        if layer.layer is not None:
            script += (
                f"    {layer_name}: Layer = ({layer.layer[0]}, {layer.layer[1]})\n"
            )

    script += """

LAYER = LayerMapFab
"""

    filepathout.write_text(script)
    return script


if __name__ == "__main__":
    from gdsfactory.config import PATH

    print(lyp_to_dataclass(PATH.klayout_lyp))
