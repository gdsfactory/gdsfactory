"""A GDS layer is a tuple of two integers.

You can:

- Define your layers in a dataclass
- Load it from Klayout XML file (.lyp)

LayerSet adapted from phidl.device_layout
load_lyp, name_to_description, name_to_short_name adapted from phidl.utilities
preview_layerset adapted from phidl.geometry
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import xmltodict
from phidl.device_layout import Layer
from phidl.device_layout import LayerSet as LayerSetPhidl

from pp.component import Component
from pp.name import clean_name


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
        """Adds a layer to an existing LayerSet object.

        Args:
            name: Name of the Layer.
            gds_layer : GDSII Layer number.
            gds_datatype : GDSII datatype.
            description : Layer description.
            color : Hex code of color for the Layer.
            inverted :  If true, inverts the Layer.
            alpha: Alpha parameter (opacity) for the Layer, value must be between 0.0 and 1.0.
            dither: KLayout dither style (only used in phidl.utilities.write_lyp() )
        """
        new_layer = Layer(
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

    def __getitem__(self, val: str) -> Tuple[int, int]:
        """Returns gds layer tuple."""
        if val not in self._layers:
            raise ValueError(f"Layer {val} not in {list(self._layers.keys())}")
        else:
            layer = self._layers[val]
            return layer.gds_layer, layer.gds_datatype

    def __repr__(self):
        """ Prints the number of Layers in the LayerSet object. """
        return (
            f"LayerSet ({len(self._layers)} layers total) \n"
            + f"{list(self._layers.keys())}"
        )

    def get(self, key: str) -> Layer:
        """Returns gds layer tuple."""
        if key not in self._layers:
            raise ValueError(f"Layer {key} not in {list(self._layers.keys())}")
        else:
            return self._layers[key]


@dataclass(frozen=True)
class LayerMap:
    WG = (1, 0)
    WGCLAD = (111, 0)
    SLAB150 = (2, 0)
    SLAB90 = (3, 0)
    WGN = (34, 0)
    N = (20, 0)
    Np = (22, 0)
    Npp = (24, 0)
    P = (21, 0)
    Pp = (23, 0)
    Ppp = (25, 0)
    HEATER = (47, 0)
    M1 = (41, 0)
    M2 = (45, 0)
    M3 = (49, 0)
    VIA1 = (40, 0)
    VIA2 = (44, 0)
    VIA3 = (43, 0)
    NO_TILE_SI = (71, 0)
    DEEPTRENCH = (7, 0)
    PADDING = (67, 0)
    DEVREC = (68, 0)
    FLOORPLAN = (64, 0)
    TEXT = (66, 0)
    PORT = (1, 10)
    PORTE = (69, 0)
    PORTH = (70, 0)
    LABEL = (201, 0)
    LABEL_SETTINGS = (202, 0)
    TE = (203, 0)
    TM = (204, 0)
    DRC_MARKER = (205, 0)
    LABEL_INSTANCE = (206, 0)


LAYER = LayerMap()

ls = LayerSet()  # Layerset makes plotgds look good
ls.add_layer("WG", LAYER.WG[0], LAYER.WG[1], "wg", color="gray", alpha=1)
ls.add_layer("WGCLAD", LAYER.WGCLAD[0], 0, "", color="gray", alpha=0)
ls.add_layer("SLAB150", LAYER.SLAB150[0], 0, "", color="lightblue", alpha=0.6)
ls.add_layer("SLAB90", LAYER.SLAB90[0], 0, "", color="lightblue", alpha=0.2)
ls.add_layer("WGN", LAYER.WGN[0], 0, "", color="orange", alpha=1)
ls.add_layer("DEVREC", LAYER.DEVREC[0], 0, "", color="gray", alpha=0.1)


layer2nm = {LAYER.WG: 220}
layer2material = {
    LAYER.WG: "si",
    LAYER.SLAB90: "si",
    LAYER.SLAB150: "si",
    LAYER.WGCLAD: "sio2",
    LAYER.WGN: "sin",
}

port_layer2type = {LAYER.PORT: "optical", LAYER.PORTE: "dc", LAYER.PORTH: "heater"}

port_type2layer = {v: k for k, v in port_layer2type.items()}


layer_cladding_waveguide = [LAYER.WGCLAD]


def preview_layerset(ls: LayerSet = ls, size: float = 100.0) -> Component:
    """Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files
    """
    import numpy as np

    import pp

    D = Component(name="layerset")
    scale = size / 100
    num_layers = len(ls._layers)
    matrix_size = int(np.ceil(np.sqrt(num_layers)))
    sorted_layers = sorted(
        ls._layers.values(), key=lambda x: (x.gds_layer, x.gds_datatype)
    )
    for n, layer in enumerate(sorted_layers):
        R = pp.c.rectangle(size=(100 * scale, 100 * scale), layer=layer)
        T = pp.c.text(
            text="%s\n%s / %s" % (layer.name, layer.gds_layer, layer.gds_datatype),
            size=20 * scale,
            position=(50 * scale, -20 * scale),
            justify="center",
            layer=layer,
        )

        xloc = n % matrix_size
        yloc = int(n // matrix_size)
        D.add_ref(R).movex(200 * xloc * scale).movey(-200 * yloc * scale)
        D.add_ref(T).movex(200 * xloc * scale).movey(-200 * yloc * scale)
    return D


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


def _name_to_description(name_str):
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


def _add_layer(entry, lys):
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

    settings = dict()
    settings["gds_layer"] = int(gds_layer)
    settings["gds_datatype"] = int(gds_datatype)
    settings["color"] = entry["fill-color"]
    settings["dither"] = entry["dither-pattern"]
    settings["name"] = _name_to_short_name(name)
    settings["description"] = _name_to_description(name)
    lys.add_layer(**settings)
    return lys


def load_lyp(filename: Path):
    """Returns a LayerSet object from a Klayout lyp file in XML format."""
    with open(filename, "r") as fx:
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


# For port labelling purpose
LAYERS_OPTICAL = [LAYER.WG]
LAYERS_ELECTRICAL = [LAYER.M1, LAYER.M2, LAYER.M3]
LAYERS_HEATER = [LAYER.HEATER]


def test_load_lyp():
    from pp.config import layer_path

    lys = load_lyp(layer_path)
    assert len(lys._layers) == 82
    return lys


if __name__ == "__main__":
    lys = test_load_lyp()

    # c = preview_layerset(ls)
    # c.show()
    # print(LAYERS_OPTICAL)
    # print(layer("wgcore"))
    # print(layer("wgclad"))
    # print(layer("padding"))
    # print(layer("TEXT"))
    # print(type(layer("wgcore")))
