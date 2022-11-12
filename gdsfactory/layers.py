"""A GDS layer is a tuple of two integers.

You can:

- Load LayerColors from Klayout XML file (.lyp) (recommended)
- Define your layers in a Pydantic BaseModel

LayerColors based on phidl.device_layout
load_lyp, name_to_description, name_to_short_name based on phidl.utilities
preview_layerset based on phidl.geometry

"""
import pathlib
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xmltodict
from pydantic import BaseModel, Field, validator

from gdsfactory.constants import _CSS3_NAMES_TO_HEX
from gdsfactory.name import clean_name

module_path = pathlib.Path(__file__).parent.absolute()
layer_path = module_path / "klayout" / "tech" / "layers.lyp"


def preview_layerset(ls, size: float = 100.0, spacing: float = 100.0) -> object:
    """Generates a Component with all the layers.

    Args:
        ls: LayerColors.
        size: square size.
        spacing: spacing between each square.

    """
    import gdsfactory as gf

    D = gf.Component(name="layerset", with_uuid=True)
    scale = size / 100
    num_layers = len(ls.layers)
    matrix_size = int(np.ceil(np.sqrt(num_layers)))
    sorted_layers = sorted(
        ls.layers.values(), key=lambda x: (x.gds_layer, x.gds_datatype)
    )
    for n, layer in enumerate(sorted_layers):
        gds_layer, gds_datatype = layer.gds_layer, layer.gds_datatype
        layer_tuple = (gds_layer, gds_datatype)
        R = gf.components.rectangle(size=(100 * scale, 100 * scale), layer=layer_tuple)
        T = gf.components.text(
            text=f"{layer.name}\n{layer.gds_layer} / {layer.gds_datatype}",
            size=20 * scale,
            position=(50 * scale, -20 * scale),
            justify="center",
            layer=layer_tuple,
        )

        xloc = n % matrix_size
        yloc = int(n // matrix_size)
        D.add_ref(R).movex((100 + spacing) * xloc * scale).movey(
            -(100 + spacing) * yloc * scale
        )
        D.add_ref(T).movex((100 + spacing) * xloc * scale).movey(
            -(100 + spacing) * yloc * scale
        )
    return D


class LayerColor(BaseModel):
    """Layer object with color, alpha (opacity) and dither.

    Parameters:
        gds_layer: int GDSII Layer number.
        gds_datatype: int GDSII datatype.
        name: str Name of the Layer.
        color: str Hex code of color for the Layer.
        alpha: Alpha parameter (opacity) for the Layer.
        dither: str KLayout dither parameter (texture) for the Layer
            (only used in phidl.utilities.write_lyp)

    """

    gds_layer: int = 0
    gds_datatype: int = 0
    name: str = "unnamed"
    description: Optional[str] = None
    inverted: bool = False
    color: Optional[str] = None
    alpha: float = 0.6
    dither: Optional[str] = None

    @validator("color")
    def color_is_valid(cls, color):
        try:
            if color is None:  # not specified
                color = None
            elif np.size(color) == 3:  # in format (0.5, 0.5, 0.5)
                color = np.array(color)
                if np.any(color > 1) or np.any(color < 0):
                    raise ValueError
                color = np.array(np.round(color * 255), dtype=int)
                color = "#{:02x}{:02x}{:02x}".format(*color)
            elif color[0] == "#":  # in format #1d2e3f
                if len(color) != 7:
                    raise ValueError
                int(color[1:], 16)  # Will throw error if not hex format
                color = color
            else:  # in named format 'gold'
                color = _CSS3_NAMES_TO_HEX[color.lower()]
        except Exception as error:
            raise ValueError(
                "LayerColor() color must be specified as a "
                "0-1 RGB triplet, (e.g. [0.5, 0.1, 0.9]), an HTML hex color string "
                "(e.g. '#a31df4'), or a CSS3 color name (e.g. 'gold' or "
                "see http://www.w3schools.com/colors/colors_names.asp )"
            ) from error
        return color


class LayerColors(BaseModel):
    """LayerColor dict.

    Parameters:
        layers: dict of LayerColors.

    """

    layers: Dict[str, LayerColor] = Field(default_factory=dict)

    def add_layer(
        self,
        name: str = "unnamed",
        gds_layer: int = 0,
        gds_datatype: int = 0,
        description: Optional[str] = None,
        color: Optional[str] = None,
        inverted: bool = False,
        alpha: float = 0.6,
        dither: Optional[str] = None,
    ) -> None:
        """Adds a layer to LayerColors object for nice colors.

        Args:
            name: Name of the Layer.
            gds_layer: GDSII Layer number.
            gds_datatype: GDSII datatype.
            description: Layer description.
            color: Hex code of color for the Layer.
            inverted: If true, inverts the Layer.
            alpha: layer opacity between 0 and 1 (0: invisible,  1: opaque).
            dither: KLayout dither style for phidl.utilities.write_lyp().

        """
        new_layer = LayerColor(
            gds_layer=gds_layer,
            gds_datatype=gds_datatype,
            name=name,
            description=description,
            inverted=inverted,
            color=color,
            alpha=alpha,
            dither=dither,
        )
        if name in self.layers:
            raise ValueError(
                f"Adding {name!r} already defined {list(self.layers.keys())}"
            )
        else:
            self.layers[name] = new_layer

    def __repr__(self):
        """Prints the number of Layers in the LayerColors object."""
        return (
            f"LayerColors ({len(self.layers)} layers total) \n"
            f"{list(self.layers.keys())}"
        )

    def get(self, name: str) -> LayerColor:
        """Returns Layer from name."""
        if name not in self.layers:
            raise ValueError(f"Layer {name!r} not in {list(self.layers.keys())}")
        else:
            return self.layers[name]

    def __getitem__(self, val):
        """Allows accessing to the layer names like ls['gold2'].

        Args:
            val: Layer name to access within the LayerColors.

        Returns:
            self.layers[val]: Accessed Layer in the LayerColors.

        """
        try:
            return self.layers[val]
        except Exception as error:
            raise ValueError(
                f"Layer {val!r} not in LayerColors {list(self.layers.keys())}"
            ) from error

    def get_from_tuple(self, layer_tuple: Tuple[int, int]) -> LayerColor:
        """Returns Layer from layer tuple (gds_layer, gds_datatype)."""
        tuple_to_name = {
            (v.gds_layer, v.gds_datatype): k for k, v in self.layers.items()
        }
        if layer_tuple not in tuple_to_name:
            raise ValueError(
                f"Layer color {layer_tuple} not in {list(tuple_to_name.keys())}"
            )

        name = tuple_to_name[layer_tuple]
        return self.layers[name]

    def get_layer_tuples(self):
        return {(layer.gds_layer, layer.gds_datatype) for layer in self.layers.values()}

    def clear(self) -> None:
        """Deletes all layers in the LayerColors."""
        self.layers = {}

    def preview(self):
        return preview_layerset(self)


def _name_to_short_name(name_str: str) -> str:
    """Maps the name entry of the lyp element to a name of the layer.

    i.e. the dictionary key used to access it.
    Default format of the lyp name are:

        - key - layer/datatype - description
        - key - description

    """
    if name_str is None:
        raise OSError(f"layer {name_str} has no name")
    fields = name_str.split("-")
    name = fields[0].split()[0].strip()
    return clean_name(name, remove_dots=True)


def _name_to_description(name_str) -> str:
    """Gets the description of the layer contained in the lyp name field.

    It is not strictly necessary to have a description. If none there, it returns ''.

    Default format of the lyp name are:

        - key - layer/datatype - description
        - key - description

    """
    if name_str is None:
        raise OSError(f"layer {name_str!r} has no name")
    fields = name_str.split()
    return " ".join(fields[1:]) if len(fields) > 1 else ""


def _add_layer(
    entry, lys: LayerColors, shorten_names: bool = True
) -> Optional[LayerColors]:
    """Adds a layer entry to layer colors and Returns a LayerColors updated dict.

    Returns None for invalid entries.

    Args:
        entry: layer entry.Entry is a dict of one element of 'properties'.
        lys: LayerColors map.
        shorten_names: if True takes the first part of the layer as its name.

    """
    info = entry["source"].split("@")[0]

    # skip layers without name or with */*
    if "'" in info or "*" in info:
        return None

    name = entry.get("name") or entry.get("source")
    if not name:
        return None

    infos = info.split("/")

    if len(infos) <= 1:
        return None

    gds_layer, gds_datatype = info.split("/")
    gds_layer = gds_layer.split()[-1]
    gds_datatype = gds_datatype.split()[-1]

    # print(entry.keys())
    # print(name, entry["xfill"], entry["fill-color"])
    # if entry["visible"] == "false" or entry["xfill"] == "false":

    name = _name_to_short_name(name) if shorten_names else name
    dither = entry["dither-pattern"]

    if ("visible" in entry.keys()) and (entry["visible"] == "false"):
        alpha = 0.0
    elif ("transparent" in entry.keys()) and (entry["transparent"] == "false"):
        alpha = 0.1 if dither == "I1" else 1.0
    else:
        alpha = 0.5

    settings = {
        "gds_layer": int(gds_layer),
        "gds_datatype": int(gds_datatype),
        "color": entry["fill-color"],
        "dither": dither,
        "name": name,
        "description": _name_to_description(name),
        "alpha": alpha,
    }

    lys.add_layer(**settings)
    return lys


def load_lyp(filepath: Path) -> LayerColors:
    """Returns a LayerColors object from a Klayout lyp file layer properties \
    file."""
    with open(filepath) as fx:
        lyp_dict = xmltodict.parse(fx.read(), process_namespaces=True)

    # lyp files have a top level that just has one dict: layer-properties
    # That has multiple children 'properties', each for a layer. So it gives a list.

    lyp_list = lyp_dict["layer-properties"]["properties"]
    if not isinstance(lyp_list, list):
        lyp_list = [lyp_list]

    lys = LayerColors()

    for entry in lyp_list:
        try:
            group_members = entry["group-members"]
        except KeyError:  # it is a real layer
            _add_layer(entry, lys)
        else:  # it is a group of other entries
            if not isinstance(group_members, list):
                group_members = [group_members]
            for member in group_members:
                try:
                    _add_layer(member, lys)
                except Exception:
                    _add_layer(member, lys, shorten_names=False)
    return lys


load_lyp_generic = partial(load_lyp, filepath=layer_path)


def lyp_to_dataclass(lyp_filepath: Union[str, Path], overwrite: bool = True) -> str:
    """Returns python LayerMap script from a klayout layer properties file lyp."""
    filepathin = pathlib.Path(lyp_filepath)
    filepathout = filepathin.with_suffix(".py")

    if filepathout.exists() and not overwrite:
        raise FileExistsError(f"You can delete {filepathout}")

    script = """
from pydantic import BaseModel
from gdsfactory.types import Layer


class LayerMap(BaseModel):
"""
    lys = load_lyp(filepathin)
    for layer_name, layer in sorted(lys.layers.items()):
        script += (
            f"    {layer_name}: Layer = ({layer.gds_layer}, {layer.gds_datatype})\n"
        )

    script += """
    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMap()
"""

    filepathout.write_text(script)
    return script


def test_load_lyp():
    from gdsfactory.config import layer_path

    lys = load_lyp(layer_path)
    assert len(lys.layers) > 10, len(lys.layers)
    return lys


try:
    LAYER_COLORS = load_lyp_generic()
except Exception:
    print(f"Error loading generic layermap in {layer_path!r}")
    LAYER_COLORS = LayerColors()


if __name__ == "__main__":
    # LAYER_COLORS = load_lyp_generic()
    # import gdsfactory as gf

    # c = gf.components.rectangle(layer=(123, 0))
    # c.plot()

    # print(LAYER_COLORS)
    # print(LAYER_STACK.get_from_tuple((1, 0)))
    # print(LAYER_STACK.get_layer_to_material())
    # layer = LayerColor(color="gold")
    # print(layer)

    # lys = test_load_lyp()
    c = preview_layerset(LAYER_COLORS)
    c.show(show_ports=True)
    # print(LAYERS_OPTICAL)
    # print(layer("wgcore"))
    # print(layer("wgclad"))
    # print(layer("padding"))
    # print(layer("TEXT"))
    # print(type(layer("wgcore")))
