import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.components.rectangle import rectangle
from gdsfactory.types import ComponentSpec, LayerSpec


@gf.cell
def pads_shorted(
    pad: ComponentSpec = pad_function,
    columns: int = 8,
    pad_spacing: float = 150.0,
    layer_metal: LayerSpec = "M3",
    metal_width: float = 10,
) -> Component:
    """Returns a 1D array of shorted_pads.

    Args:
        pad: pad spec.
        columns: number of columns.
        pad_spacing: in um
        layer_metal: for the short.
        metal_width: for the short.
    """
    c = Component()
    pad = gf.get_component(pad)
    for i in range(columns):
        pad_ref = c.add_ref(pad)
        pad_ref.movex(i * pad_spacing - columns / 2 * pad_spacing + pad_spacing / 2)

    short = rectangle(
        size=(pad_spacing * (columns - 1), metal_width),
        layer=layer_metal,
        centered=True,
    )
    c.add_ref(short)
    return c


if __name__ == "__main__":

    c = pads_shorted(metal_width=20)
    c.show(show_ports=True)
