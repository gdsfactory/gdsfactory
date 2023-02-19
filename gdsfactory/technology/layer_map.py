import pathlib
from typing import Union

from gdsfactory.technology.layer_views import LayerViews


def lyp_to_dataclass(
    lyp_filepath: Union[str, pathlib.Path], overwrite: bool = True
) -> str:
    """Returns python LayerMap script from a klayout layer properties file lyp."""
    filepathin = pathlib.Path(lyp_filepath)
    filepathout = filepathin.with_suffix(".py")

    if filepathout.exists() and not overwrite:
        raise FileExistsError(f"You can delete {filepathout}")

    script = """
from pydantic import BaseModel
from gdsfactory.typings import Layer


class LayerMap(BaseModel):
"""
    lys = LayerViews.from_lyp(filepathin)
    for layer_name, layer in sorted(lys.get_layer_views().items()):
        script += f"    {layer_name}: Layer = ({layer.layer[0]}, {layer.layer[1]})\n"

    script += """
    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMap()
"""

    filepathout.write_text(script)
    return script


if __name__ == "__main__":
    from gdsfactory.config import PATH

    print(lyp_to_dataclass(PATH.klayout_lyp))
