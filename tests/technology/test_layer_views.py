import gdsfactory as gf
from gdsfactory import partial
from gdsfactory.technology import LayerStack, LayerView, LayerViews
from gdsfactory.typings import Layer, LayerLevel, LayerMap

nm = 1e-3


class LayerMapFabA(LayerMap):
    WG: Layer = (34, 0)
    SLAB150: Layer = (2, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TEXT: Layer = (66, 0)


LAYER = LayerMapFabA


class FabALayerViews(LayerViews):
    WG: LayerView = LayerView(color="gold")
    SLAB150: LayerView = LayerView(color="red")
    TE: LayerView = LayerView(color="green")


LAYER_VIEWS = FabALayerViews(layers=LAYER)


def get_layer_stack_faba(
    thickness_wg: float = 500 * nm, thickness_slab: float = 150 * nm
) -> LayerStack:
    """Returns fabA LayerStack."""
    return LayerStack(
        layers=dict(
            strip=LayerLevel(
                layer=LAYER.WG,
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
            ),
            strip2=LayerLevel(
                layer=LAYER.SLAB150,
                thickness=thickness_slab,
                zmin=0.0,
                material="si",
            ),
        )
    )


if __name__ == "__main__":
    LAYER_STACK = get_layer_stack_faba()
    WIDTH = 2

    # Specify a cross_section to use
    strip = partial(gf.cross_section.cross_section, width=WIDTH, layer=LAYER.WG)

    mmi1x2 = partial(
        gf.components.mmi1x2,
        width=WIDTH,
        width_taper=WIDTH,
        width_mmi=3 * WIDTH,
        cross_section=strip,
    )

    PDK = gf.Pdk(
        name="Fab_A",
        cells=dict(mmi1x2=mmi1x2),
        cross_sections=dict(strip=strip),
        layers=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    PDK.activate()
    PDK_A = PDK

    # gc = partial(
    #     gf.components.grating_coupler_elliptical_te, layer=LAYER.WG, cross_section=strip
    # )

    # c = gf.components.mzi()
    # c_gc = gf.routing.add_fiber_array(
    #     component=c, grating_coupler=gc, with_loopback=False
    # )
    # c_gc.show()
