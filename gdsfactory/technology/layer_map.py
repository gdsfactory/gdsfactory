import pathlib
from collections.abc import Iterable

import gdsfactory as gf
from gdsfactory.technology.layer_views import LayerViews


class LayerMap(gf.LayerEnum):
    """You will need to create a new LayerMap with your specific foundry layers."""

    kcl = gf.constant(gf.kcl)

    @classmethod
    def to_dict(cls) -> dict[str, tuple[int, int]]:
        layer_dict = {}
        for attribute_name in dir(cls):
            value = getattr(cls, attribute_name)
            if isinstance(value, Iterable) and len(value) == 2:
                layer_dict[attribute_name] = (value[0], value[1])
        return layer_dict


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
        script += f"    {layer_name}: Layer = ({layer.layer[0]}, {layer.layer[1]})\n"

    script += """

LAYER = LayerMapFab()
"""

    filepathout.write_text(script)
    return script


if __name__ == "__main__":
    layers = LayerMap
    # from gdsfactory.config import PATH

    # print(lyp_to_dataclass(PATH.klayout_lyp))
