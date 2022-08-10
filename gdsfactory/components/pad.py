from functools import partial
from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.types import ComponentSpec, LayerSpec


@cell
def pad(
    size: Tuple[float, float] = (100.0, 100.0),
    layer: LayerSpec = "M3",
    bbox_layers: Optional[Tuple[LayerSpec, ...]] = None,
    bbox_offsets: Optional[Tuple[float, ...]] = None,
    port_inclusion: float = 0,
    port_orientation: Optional[float] = None,
) -> Component:
    """Returns rectangular pad with 4 ports (1, 2, 3, 4).

    Args:
        size: x, y size.
        layer: pad layer.
        bbox_layers: list of layers.
        bbox_offsets: list of offsets.
        port_inclusion: from edge.
        port_orientation: in degrees.
    """
    c = Component()
    layer = gf.get_layer(layer)
    rect = compass(size=size, layer=layer, port_inclusion=port_inclusion)
    c_ref = c.add_ref(rect)
    c.add_ports(c_ref.ports)
    c.info["size"] = (float(size[0]), float(size[1]))
    c.info["layer"] = layer

    if bbox_layers and bbox_offsets:
        sizes = []
        for cladding_offset in bbox_offsets:
            size = (size[0] + 2 * cladding_offset, size[1] + 2 * cladding_offset)
            sizes.append(size)

        for layer, size in zip(bbox_layers, sizes):
            c.add_ref(
                compass(
                    size=size,
                    layer=layer,
                )
            )

    width = size[1] if port_orientation in [0, 180] else size[0]

    c.add_port(
        name="pad",
        port_type="vertical_dc",
        layer=layer,
        center=(0, 0),
        orientation=port_orientation,
        width=width,
    )
    return c


@cell
def pad_array(
    pad: ComponentSpec = "pad",
    spacing: Tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
    orientation: Optional[float] = 270,
) -> Component:
    """Returns 2D array of pads.

    Args:
        pad: pad element.
        spacing: x, y pitch.
        columns: number of columns.
        rows: number of rows.
        orientation: port orientation in deg. None for low speed DC ports.
    """
    c = Component()
    pad = gf.get_component(pad)
    size = pad.settings.full["size"]
    c.info["size"] = size

    c.add_array(pad, columns=columns, rows=rows, spacing=spacing)
    width = size[0] if orientation in [90, 270] else size[1]

    for col in range(columns):
        for row in range(rows):
            c.add_port(
                name=f"e{row+1}{col+1}",
                center=(col * spacing[0], row * spacing[1]),
                width=width,
                orientation=orientation,
                port_type="electrical",
                layer=pad.info["layer"],
            )
    return c


pad_array90 = partial(pad_array, orientation=90)
pad_array270 = partial(pad_array, orientation=270)

pad_array0 = partial(pad_array, orientation=0, columns=1, rows=3)
pad_array180 = partial(pad_array, orientation=180, columns=1, rows=3)


if __name__ == "__main__":
    c = pad()
    # c = pad(layer_to_inclusion={(3, 0): 10})
    # print(c.ports)
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # c = pad_array90()
    # c = pad_array0()
    # c = pad_array270()
    # c.pprint_ports()
    # c = pad_array_2d(cols=2, rows=3, port_names=("e2",))
    # c = pad_array(columns=2, rows=2, orientation=270)
    # c.auto_rename_ports()
    c.show(show_ports=True)
