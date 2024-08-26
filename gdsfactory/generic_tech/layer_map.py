import gdsfactory as gf
from gdsfactory.technology.layer_map import LayerInfo, LayerMap


class LayerClass(LayerMap):
    """Generic layermap based on book.

    Lukas Chrostowski, Michael Hochberg, "Silicon Photonics Design",
    Cambridge University Press 2015, page 353
    You will need to create a new LayerInfoMap with your specific foundry layers.
    """

    WAFER: LayerInfo = LayerInfo(999, 0)
    WG: LayerInfo = LayerInfo(1, 0)
    WGCLAD: LayerInfo = LayerInfo(111, 0)
    SLAB150: LayerInfo = LayerInfo(2, 0)
    SHALLOW_ETCH: LayerInfo = LayerInfo(2, 6)
    SLAB90: LayerInfo = LayerInfo(3, 0)
    DEEP_ETCH: LayerInfo = LayerInfo(3, 6)
    DEEPTRENCH: LayerInfo = LayerInfo(4, 0)
    GE: LayerInfo = LayerInfo(5, 0)
    UNDERCUT: LayerInfo = LayerInfo(6, 0)
    WGN: LayerInfo = LayerInfo(34, 0)
    WGN_CLAD: LayerInfo = LayerInfo(36, 0)

    N: LayerInfo = LayerInfo(20, 0)
    NP: LayerInfo = LayerInfo(22, 0)
    NPP: LayerInfo = LayerInfo(24, 0)
    P: LayerInfo = LayerInfo(21, 0)
    PP: LayerInfo = LayerInfo(23, 0)
    PPP: LayerInfo = LayerInfo(25, 0)
    GEN: LayerInfo = LayerInfo(26, 0)
    GEP: LayerInfo = LayerInfo(27, 0)

    HEATER: LayerInfo = LayerInfo(47, 0)
    M1: LayerInfo = LayerInfo(41, 0)
    M2: LayerInfo = LayerInfo(45, 0)
    M3: LayerInfo = LayerInfo(49, 0)
    MTOP: LayerInfo = LayerInfo(49, 0)
    VIAC: LayerInfo = LayerInfo(40, 0)
    VIA1: LayerInfo = LayerInfo(44, 0)
    VIA2: LayerInfo = LayerInfo(43, 0)
    PADOPEN: LayerInfo = LayerInfo(46, 0)

    DICING: LayerInfo = LayerInfo(100, 0)
    NO_TILE_SI: LayerInfo = LayerInfo(71, 0)
    PADDING: LayerInfo = LayerInfo(67, 0)
    DEVREC: LayerInfo = LayerInfo(68, 0)
    FLOORPLAN: LayerInfo = LayerInfo(64, 0)
    TEXT: LayerInfo = LayerInfo(66, 0)
    PORT: LayerInfo = LayerInfo(1, 10)
    WG_PIN: LayerInfo = LayerInfo(1, 10)
    PORTE: LayerInfo = LayerInfo(1, 11)
    PORTH: LayerInfo = LayerInfo(70, 0)
    SHOW_PORTS: LayerInfo = LayerInfo(1, 12)
    LABEL_INSTANCE: LayerInfo = LayerInfo(206, 0)
    LABEL_SETTINGS: LayerInfo = LayerInfo(202, 0)
    TE: LayerInfo = LayerInfo(203, 0)
    TM: LayerInfo = LayerInfo(204, 0)
    DRC_MARKER: LayerInfo = LayerInfo(205, 0)

    SOURCE: LayerInfo = LayerInfo(110, 0)
    MONITOR: LayerInfo = LayerInfo(101, 0)


LAYER = LayerClass()
gf.kcl.infos = LAYER


if __name__ == "__main__":
    LAYER.my_layer = LayerInfo(1, 2)
