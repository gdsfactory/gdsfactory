import pathlib

import gdsfactory as gf
from gdsfactory.technology.layer_views import LayerViews


class LayerMap(gf.LayerEnum):
    """You will need to create a new LayerMap with your specific foundry layers."""

    layout = gf.constant(gf.kcl.layout)


def lyp_to_dataclass(
    lyp_filepath: str | pathlib.Path,
    overwrite: bool = True,
    sort_by_name: bool = True,  # sort_by_name=False means same order as in lyp file
    map_name: str = "LayerMapFab",
    output_filepath: pathlib.Path | str | None = None,
) -> str:
    """Returns python LayerMap script from a klayout layer properties file lyp."""
    filepathin = pathlib.Path(lyp_filepath)

    if output_filepath is None:
        filepathout = filepathin.with_suffix(".py")
    else:
        filepathout = pathlib.Path(output_filepath)
    filepathout.parent.mkdir(parents=True, exist_ok=True)

    if filepathout.exists() and not overwrite:
        raise FileExistsError(f"You can delete {filepathout}")

    if not map_name.isidentifier():
        raise ValueError(
            f"Argument 'map_name' must be a valid python identifier, but {map_name} is not."
        )

    script = f"""from gdsfactory.typings import Layer
from gdsfactory.technology.layer_map import LayerMap


class {map_name}(LayerMap):
"""
    lys = LayerViews.from_lyp(filepathin)
    maybe_sort = sorted if sort_by_name else lambda seq: seq
    for layer_name, layer in maybe_sort(lys.get_layer_views().items()):
        if layer.layer is not None:
            script += (
                f"    {layer_name}: Layer = ({layer.layer[0]}, {layer.layer[1]})\n"
            )

    script += f"""

LAYER = {map_name}
"""

    filepathout.write_text(script)
    return script


if __name__ == "__main__":
    from gdsfactory.config import PATH

    print(lyp_to_dataclass(PATH.klayout_lyp))
