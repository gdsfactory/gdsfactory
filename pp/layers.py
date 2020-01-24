""" there are multiple options to define layers:

- using an int
- using `pp.LAYER` object, with a layer per property
- using `pp.layer(name, layermap)` function

"""

from collections import namedtuple
from phidl import LayerSet

import gdspy as gp
from phidl.device_layout import DeviceReference
from phidl.device_layout import Polygon


generic_layermap = dict(
    WG=1,
    SLAB150=2,
    SLAB90=3,
    DEEPTRENCH=7,
    WGN=34,
    HEATER=47,
    M1=41,
    M2=45,
    M3=49,
    VIA1=40,
    VIA2=44,
    VIA3=43,
    NBN=31,
    TEXT=66,
    PORT=60,
    NO_TILE_SI=63,
    FLOORPLAN=600,
    FLOORPLAN_PACKAGING=601,
    FLOORPLAN_WIREBOND_LANE=602,
    FLOORPLAN_SI_REMOVAL=603,
    FLOORPLAN_PACKAGING_OPTICAL=604,
    FLOORPLAN_E_DIE=610,
    FLOORPLAN_E_DIE_COMPONENTS=611,
    FLOORPLAN_CU_HEAT_SINK=620,
    LABEL=201,
    INFO_GEO_HASH=202,
    polarization_te=203,
    polarization_tm=204,
)


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

layer2m = {1: 220e-9, 2: 150e-9, 3: 90 - 3}
layer2material = {
    1: "Si (Silicon) - Palik",
    2: "SiO2 (Glass) - Palik",
    3: "Si (Silicon) - Palik",
    34: "Si3N4 (Silicon Nitride) - Phillip",
}

LAYER = namedtuple("layer", generic_layermap.keys())(*generic_layermap.values())
generic_layermap.update(ls._layers)


def layer(name, layermap=generic_layermap):
    """ returns the gds layer number from layermap dictionary"""
    layer = layermap.get(name)
    if layer:
        if isinstance(layer, int):
            return layer
        return layer.gds_layer
    else:
        raise ValueError(
            "{} is not a valid layer_name. Valid names are: \n{}".format(
                name, "\n".join(layermap.keys())
            )
        )


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

    .. plot::
      :include-source:

      import pp
      c = pp.preview_layerset()
      pp.plotgds(c)
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
LAYERS_OPTICAL = [layer("wgcore")]
LAYERS_ELECTRICAL = [layer("m1"), layer("m2"), layer("m3")]
LAYERS_HEATER = [layer("mh")]
LAYERS_SUPERCONDUCTING = [layer("nbn")]

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
