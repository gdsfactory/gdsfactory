""" there are multiple options to define layers:

- using an int
- using `pp.LAYER` object, with a layer per property
- using `pp.layer(name, layermap)` function

"""

from dataclasses import dataclass
import gdspy as gp
from phidl import LayerSet
from phidl.device_layout import DeviceReference
from phidl.device_layout import Polygon


@dataclass
class Layer:
    WG = (1, 0)
    WGCLAD = (111, 0)
    SLAB150 = (2, 0)
    SLAB90 = (3, 0)
    DEEPTRENCH = (7, 0)
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
    NO_TILE_SI = (63, 0)
    PADDING = (67, 0)
    DEVREC = (68, 0)
    FLOORPLAN = (600, 0)
    TEXT = (66, 0)
    PORT = (1, 10)
    PORTE = (69, 0)
    PORTH = (70, 0)
    LABEL = (201, 0)
    INFO_GEO_HASH = (202, 0)
    polarization_te = (203, 0)
    polarization_tm = (204, 0)


LAYER = Layer()

# This is only for plotgds to look good
ls = LayerSet()  # Create a blank LayerSet
ls.add_layer("wgcore", 1, 0, "wgcore", "gray")
ls.add_layer("wgclad", 111, 0, "wgclad", "gray", alpha=0)
ls.add_layer("slab150", 2, 0, "slab150", "lightblue", alpha=0.2)
ls.add_layer("slab150clad", 112, 0, "slab150clad", "lightblue", alpha=0.2)
ls.add_layer("slab150trench", 122, 0, "slab150trench", "lightblue", alpha=0.2)
ls.add_layer("slab90", 3, 0, "slab90", "gray", alpha=0.5)
ls.add_layer("slab90clad", 113, 0, "slab90clad", "gray", alpha=0.5)
ls.add_layer("slab90trench", 123, 0, "slab90trench", "gray", alpha=0.5)
ls.add_layer("wgn", 34, 0, "wgn", "orange")
ls.add_layer("wgnclad", 36, 0, "wgn_clad", "gray", alpha=0)

ls.add_layer("n", 20, 0, "n", "red", alpha=0.2)
ls.add_layer("np", 22, 0, "np", "red", alpha=0.4)
ls.add_layer("npp", 24, 0, "npp", "red", alpha=0.6)

ls.add_layer("p", 21, 0, "p", "blue", alpha=0.2)
ls.add_layer("pp", 23, 0, "pp", "blue", alpha=0.4)
ls.add_layer("ppp", 25, 0, "ppp", "blue", alpha=0.6)

ls.add_layer("nbn", 31, 0, "nbn", "green", alpha=0.2)

ls.add_layer("m1", 41, 0, "m1", "green", alpha=0.2)
ls.add_layer("m2", 45, 0, "m2", "green", alpha=0.4)
ls.add_layer("m3", 49, 0, "m3", "green", alpha=0.8)
ls.add_layer("mh", 47, 0, "mh", "orange", alpha=0.2)

ls.add_layer("via1", 40, 0, "via1", "red", alpha=0.2)
ls.add_layer("via2", 40, 0, "via2", "red", alpha=0.2)
ls.add_layer("via3", 40, 0, "via3", "red", alpha=0.2)

ls.add_layer("txt", 66, 0, "txt", "grey", alpha=0.5)
ls.add_layer("label", 201, 0, "label", "grey", alpha=0)
ls.add_layer("geo_hash", 202, 0, "label", "grey", alpha=0)
ls.add_layer("te", 203, 0, "label", "grey", alpha=0)
ls.add_layer("tm", 204, 0, "label", "grey", alpha=0)
ls.add_layer("floorplan", 64, 0, "floorplan", "grey", alpha=0.2)
ls.add_layer("dicing", 65, 0, "dicing", "grey", alpha=0.2)
ls.add_layer("padding", 68, 0, "padding", "grey", alpha=0.2)
ls.add_layer("floorplan_old", 99, 0, "floorplan_old", "grey", alpha=0)
ls.add_layer("devrec", 68, 0, "devrec", "grey", alpha=0)
ls.add_layer("drcexclude", 67, 0, "drcexclude", "grey", alpha=0)
ls.add_layer("pin", 1, 10, "pin", "gray", alpha=0)
ls.add_layer("no_tile_si", 63, 0, "no_tile_si", "grey", alpha=0)
ls.add_layer("no_tile_m1", 41, 30, "no_tile_m1", "gray", alpha=0)
ls.add_layer("no_tile_m2", 45, 30, "no_tile_m2", "gray", alpha=0)
ls.add_layer("no_tile_m3", 49, 30, "no_tile_m3", "gray", alpha=0)

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


def get_gds_layers(device):
    """ Returns a set of layers in this cell.

    Returns:
        out : Set of the layers used in this cell.
    """
    layers = set()
    for element in device.references:
        if isinstance(element, Polygon):
            gds_layer = (element.layers[0], element.datatypes[0])
            layers.update([gds_layer])
        elif isinstance(element, DeviceReference) or isinstance(element, gp.CellArray):
            layers.update(get_gds_layers(element.ref_cell))
    for label in device.labels:
        datatype = label.datatype if hasattr(label, "datatype") else 0
        layers.update([(label.layer, datatype)])
    return layers


def preview_layerset(ls=ls, size=100):
    """ Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files
    """
    import pp
    import numpy as np

    D = pp.Component(name="layerset")
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


# For port labelling purpose
LAYERS_OPTICAL = [LAYER.WG]
LAYERS_ELECTRICAL = [LAYER.M1, LAYER.M2, LAYER.M3]
LAYERS_HEATER = [LAYER.HEATER]

if __name__ == "__main__":
    import pp

    c = preview_layerset(ls)
    pp.show(c)
    # print(LAYERS_OPTICAL)
    # print(layer("wgcore"))
    # print(layer("wgclad"))
    # print(layer("padding"))
    # print(layer("TEXT"))
    # print(type(layer("wgcore")))
