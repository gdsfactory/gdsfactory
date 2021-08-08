"""A GDS layer is a tuple of two integers.

You can:

- Define your layers in a dataclass
- Load it from Klayout XML file (.lyp)

LayerSet adapted from phidl.device_layout
load_lyp, name_to_description, name_to_short_name adapted from phidl.utilities
preview_layerset adapted from phidl.geometry
"""
import pathlib
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import xmltodict
from phidl.device_layout import Layer as LayerPhidl
from phidl.device_layout import LayerSet as LayerSetPhidl

from gdsfactory.name import clean_name

module_path = pathlib.Path(__file__).parent.absolute()
layer_path = module_path / "klayout" / "tech" / "layers.lyp"


def preview_layerset(
    ls: LayerSetPhidl, size: float = 100.0, spacing: float = 100.0
) -> object:
    """Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files

    Args:
        ls: LayerSet

    """
    import numpy as np

    import gdsfactory as gf

    D = gf.Component(name="layerset")
    scale = size / 100
    num_layers = len(ls._layers)
    matrix_size = int(np.ceil(np.sqrt(num_layers)))
    sorted_layers = sorted(
        ls._layers.values(), key=lambda x: (x.gds_layer, x.gds_datatype)
    )
    for n, layer in enumerate(sorted_layers):
        gds_layer, gds_datatype = layer.gds_layer, layer.gds_datatype
        layer_tuple = (gds_layer, gds_datatype)
        R = gf.components.rectangle(size=(100 * scale, 100 * scale), layer=layer_tuple)
        T = gf.components.text(
            text="%s\n%s / %s" % (layer.name, layer.gds_layer, layer.gds_datatype),
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


class LayerSet(LayerSetPhidl):
    def add_layer(
        self,
        name: str = "unnamed",
        gds_layer: int = 0,
        gds_datatype: int = 0,
        description: Optional[str] = None,
        color: Optional[str] = None,
        inverted: bool = False,
        alpha: float = 0.6,
        dither: bool = None,
    ):
        """Adds a layer to an existing LayerSet object for nice colors.

        Args:
            name: Name of the Layer.
            gds_layer: GDSII Layer number.
            gds_datatype: GDSII datatype.
            description: Layer description.
            color: Hex code of color for the Layer.
            inverted: If true, inverts the Layer.
            alpha: layer opacity between 0 and 1.
            dither: KLayout dither style, only used in phidl.utilities.write_lyp().
        """
        new_layer = LayerPhidl(
            gds_layer=gds_layer,
            gds_datatype=gds_datatype,
            name=name,
            description=description,
            inverted=inverted,
            color=color,
            alpha=alpha,
            dither=dither,
        )
        if name in self._layers:
            raise ValueError(
                f"Adding {name} already defined {list(self._layers.keys())}"
            )
        else:
            self._layers[name] = new_layer

    def __repr__(self):
        """Prints the number of Layers in the LayerSet object."""
        return (
            f"LayerSet ({len(self._layers)} layers total) \n"
            + f"{list(self._layers.keys())}"
        )

    def get(self, name: str) -> LayerPhidl:
        """Returns Layer from name."""
        if name not in self._layers:
            raise ValueError(f"Layer {name} not in {list(self._layers.keys())}")
        else:
            return self._layers[name]

    def get_from_tuple(self, layer_tuple: Tuple[int, int]) -> LayerPhidl:
        """Returns Layer from layer tuple (gds_layer, gds_datatype)."""
        tuple_to_name = {
            (v.gds_layer, v.gds_datatype): k for k, v in self._layers.items()
        }
        if layer_tuple not in tuple_to_name:
            raise ValueError(f"Layer {layer_tuple} not in {list(tuple_to_name.keys())}")

        name = tuple_to_name[layer_tuple]
        return self._layers[name]

    def clear(self):
        """Deletes all entries in the LayerSet"""
        self._layers = {}

    def preview(self):
        return preview_layerset(self)


def _name_to_short_name(name_str: str) -> str:
    """Maps the name entry of the lyp element to a name of the layer,
    i.e. the dictionary key used to access it.
    Default format of the lyp name is
        key - layer/datatype - description
        or
        key - description

    """
    if name_str is None:
        raise IOError(f"layer {name_str} has no name")
    fields = name_str.split("-")
    name = fields[0].split()[0].strip()
    return clean_name(name)


def _name_to_description(name_str) -> str:
    """Gets the description of the layer contained in the lyp name field.
    It is not strictly necessary to have a description. If none there, it returns ''.

    Default format of the lyp name is
        key - layer/datatype - description
        or
        key - description

    """
    if name_str is None:
        raise IOError(f"layer {name_str} has no name")
    fields = name_str.split()
    description = ""
    if len(fields) > 1:
        description = " ".join(fields[1:])
    return description


def _add_layer(entry, lys: LayerSet) -> LayerSet:
    """Entry is a dict of one element of 'properties'.
    No return value. It adds it to the lys variable directly
    """
    info = entry["source"].split("@")[0]

    # skip layers without name or with */*
    if "'" in info or "*" in info:
        return

    name = entry.get("name") or entry.get("source")
    if not name:
        return

    gds_layer, gds_datatype = info.split("/")
    gds_layer = gds_layer.split()[-1]
    gds_datatype = gds_datatype.split()[-1]

    # print(entry.keys())
    # print(name, entry["xfill"], entry["fill-color"])
    # if entry["visible"] == "false" or entry["xfill"] == "false":

    if entry["visible"] == "false":
        alpha = 0
    elif entry["transparent"] == "false":
        alpha = 1
    else:
        alpha = 0.5

    settings = dict()
    settings["gds_layer"] = int(gds_layer)
    settings["gds_datatype"] = int(gds_datatype)
    settings["color"] = entry["fill-color"]
    settings["dither"] = entry["dither-pattern"]
    settings["name"] = _name_to_short_name(name)
    settings["description"] = _name_to_description(name)
    settings["alpha"] = alpha
    lys.add_layer(**settings)
    return lys


def load_lyp(filepath: Path) -> LayerSet:
    """Returns a LayerSet object from a Klayout lyp file in XML format."""
    with open(filepath, "r") as fx:
        lyp_dict = xmltodict.parse(fx.read(), process_namespaces=True)
    # lyp files have a top level that just has one dict: layer-properties
    # That has multiple children 'properties', each for a layer. So it gives a list
    lyp_list = lyp_dict["layer-properties"]["properties"]
    if not isinstance(lyp_list, list):
        lyp_list = [lyp_list]

    lys = LayerSet()

    for entry in lyp_list:
        try:
            group_members = entry["group-members"]
        except KeyError:  # it is a real layer
            _add_layer(entry, lys)
        else:  # it is a group of other entries
            if not isinstance(group_members, list):
                group_members = [group_members]
            for member in group_members:
                _add_layer(member, lys)
    return lys


load_lyp_generic = partial(load_lyp, filepath=layer_path)


def lyp_to_dataclass(lyp_filepath: Union[str, Path], overwrite: bool = True) -> str:
    filepathin = pathlib.Path(lyp_filepath)
    filepathout = filepathin.with_suffix(".py")

    if filepathout.exists() and not overwrite:
        raise FileExistsError(f"You can delete {filepathout}")

    script = """
import dataclasses

@dataclasses.dataclass
class LayerMap():
"""
    lys = load_lyp(filepathin)
    for layer_name, layer in sorted(lys._layers.items()):
        script += (
            f"    {layer_name}: Layer = ({layer.gds_layer}, {layer.gds_datatype})\n"
        )

    filepathout.write_text(script)
    return script


def test_load_lyp():
    from gdsfactory.config import layer_path

    lys = load_lyp(layer_path)
    assert len(lys._layers) == 82
    return lys


if __name__ == "__main__":
    # print(LAYER_STACK.get_from_tuple((1, 0)))
    # print(LAYER_STACK.get_layer_to_material())

    # lys = test_load_lyp()
    lys = load_lyp_generic()
    c = preview_layerset(ls=lys)
    c.show()
    # print(LAYERS_OPTICAL)
    # print(layer("wgcore"))
    # print(layer("wgclad"))
    # print(layer("padding"))
    # print(layer("TEXT"))
    # print(type(layer("wgcore")))
