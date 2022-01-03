from functools import partial
from typing import Optional, Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Layer


@cell
def pad(
    size: Tuple[float, float] = (100.0, 100.0),
    layer: Layer = LAYER.M3,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    cladding_offsets: Optional[Tuple[float, ...]] = None,
    port_inclusion: float = 0,
) -> Component:
    """Rectangular pad with 4 ports (1, 2, 3, 4)

    Args:
        size:
        layer: pad layer
        layers_cladding:
        cladding_offsets:
        port_inclusion: from edge
    """
    c = Component()
    rect = compass(size=size, layer=layer, port_inclusion=port_inclusion)
    c_ref = c.add_ref(rect)
    c.add_ports(c_ref.ports)
    c.info.size = (float(size[0]), float(size[1]))
    c.info.layer = layer

    if layers_cladding and cladding_offsets:
        for layer, cladding_offset in zip(layers_cladding, cladding_offsets):
            c.add_ref(
                compass(
                    size=(size[0] + 2 * cladding_offset, size[1] + 2 * cladding_offset),
                    layer=layer,
                )
            )

    c.add_port(name="pad", port_type="vertical_dc", layer=layer, orientation=0)
    return c


@cell
def pad_array(
    pad: ComponentOrFactory = pad,
    spacing: Tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
    orientation: int = 270,
) -> Component:
    """Returns 2D array of pads

    Args:
        pad: pad element
        spacing: x, y pitch
        columns:
        rows:
        orientation: port orientation in deg
    """
    c = Component()
    pad = pad() if callable(pad) else pad
    size = pad.info.full.size
    c.info.size = size

    c.add_array(pad, columns=columns, rows=rows, spacing=spacing)
    width = size[0] if orientation in [90, 270] else size[1]

    for col in range(columns):
        for row in range(rows):
            c.add_port(
                name=f"e{row+1}{col+1}",
                midpoint=(col * spacing[0], row * spacing[1]),
                width=width,
                orientation=orientation,
                port_type="electrical",
                layer=pad.info["layer"],
            )
    return c


pad_array90 = partial(pad_array, orientation=90)
pad_array270 = partial(pad_array, orientation=270)


if __name__ == "__main__":
    c = pad()
    # c = pad(layer_to_inclusion={(3, 0): 10})
    # print(c.ports)
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # c = pad_array90()
    # c = pad_array270()
    # c.pprint_ports()
    # c = pad_array_2d(cols=2, rows=3, port_names=("e2",))
    # c = pad_array(columns=2, rows=2, orientation=270)
    # c.auto_rename_ports()
    c.show()
