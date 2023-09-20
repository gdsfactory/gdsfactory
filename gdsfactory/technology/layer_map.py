import pathlib
from typing import Any

from pydantic import BaseModel, model_validator

from gdsfactory.technology.layer_views import LayerViews

# required_layers = ["TEXT"]
required_layers = []


class LayerMap(BaseModel):
    """You will need to create a new LayerMap with your specific foundry layers."""

    model_config = {"frozen": True}

    @model_validator(mode="after")
    @classmethod
    def check_required_layers(cls, data: Any) -> Any:
        for layer in required_layers:
            if layer not in data.model_fields.keys():
                raise ValueError(f"{layer} layer must be defined")
        return data

    @model_validator(mode="after")
    @classmethod
    def check_all_layers_are_tuples_of_int(cls, data: Any) -> Any:
        for key, layer in data.model_fields.items():
            layer = layer.default
            if (
                not isinstance(layer, tuple)
                or len(layer) != 2
                or not all(isinstance(x, int) for x in layer)
            ):
                raise TypeError(f"{key} = {layer} must be a tuple of two integers")
        return data


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
    layers = LayerMap()
    # from gdsfactory.config import PATH

    # print(lyp_to_dataclass(PATH.klayout_lyp))
