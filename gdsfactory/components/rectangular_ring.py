from gdsfactory.component import Component
from gdsfactory.geometry import boolean
from gdsfactory.typings import LayerSpec
from gdsfactory.components.rectangle import rectangle
import gdsfactory as gf


@gf.cell
def rectangular_ring(
    enclosed_size=(4.0, 2.0),
    width: float = 0.5,
    layer: LayerSpec = "WG",
    centered: bool = False,
) -> Component:
    """Returns a Rectangular Ring

    Args:
        enclosed_size: (width, height) of the enclosed area.
        width: width of the ring.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0,0), False sets south-west to (0,0).
    """
    rect_in = rectangle(size=enclosed_size, centered=centered, layer=layer).ref()
    rect_out = rectangle(
        size=[dim + 2 * width for dim in enclosed_size], centered=centered, layer=layer
    )
    if not centered:
        rect_in.move((width, width))
    return boolean(A=rect_out, B=rect_in, operation="A-B", layer=layer)


if __name__ == "__main__":
    c = rectangular_ring(centered=True)
    c.show()
