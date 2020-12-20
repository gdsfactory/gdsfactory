""" there are multiple options to define layers:

- using an int
- using `pp.LAYER` object, with a layer per property
- using `pp.layer(name, layermap)` function

"""

from dataclasses import dataclass

from phidl.device_layout import Layer
from phidl.device_layout import LayerSet as LayerSetPhidl


class LayerSet(LayerSetPhidl):
    def add_layer(
        self,
        name="unnamed",
        gds_layer=0,
        gds_datatype=0,
        description=None,
        color=None,
        inverted=False,
        alpha=0,
        dither=None,
    ):
        """Adds a layer to an existing LayerSet object.

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
                '[PHIDL] LayerSet: Tried to add layer named "%s"' % (name)
                + ", but a layer with that name already exists in this LayerSet"
            )
        else:
            self._layers[name] = new_layer


@dataclass
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
    NO_TILE_SI = (63, 0)
    DEEPTRENCH = (7, 0)
    PADDING = (67, 0)
    DEVREC = (68, 0)
    FLOORPLAN = (600, 0)
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


def preview_layerset(ls=ls, size=100):
    """Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files
    """
    import numpy as np

    import pp

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
