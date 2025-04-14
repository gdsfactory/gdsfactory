from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Floats, LayerSpecs, Size


@gf.cell_with_module_name
def resistance_sheet(
    width: float = 10.0,
    layers: LayerSpecs = ("HEATER",),
    layer_offsets: Floats = (0, 0.2),
    pad: ComponentSpec = "via_stack_heater_mtop",
    pad_size: Size = (50.0, 50.0),
    pad_pitch: float = 100.0,
    ohms_per_square: float | None = None,
    pad_port_name: str = "e4",
) -> Component:
    """Returns Sheet resistance.

    keeps connectivity for pads and first layer in layers

    Args:
        width: in um.
        layers: for the middle part.
        layer_offsets: from edge, positive: over, negative: inclusion.
        pad: function to create a pad.
        pad_size: in um.
        pad_pitch: in um.
        ohms_per_square: optional sheet resistance to compute info.resistance.
        pad_port_name: port name for the pad.
    """
    c = Component()

    pad = gf.get_component(pad, size=pad_size)
    length = pad_pitch - pad_size[0]

    pad1 = c << pad
    pad2 = c << pad
    r0 = c << gf.c.compass(
        size=(length + layer_offsets[0], width + layer_offsets[0]), layer=layers[0]
    )

    for layer, offset in zip(layers[1:], layer_offsets[1:]):
        _ = c << gf.c.compass(
            size=(length + 2 * offset, width + 2 * offset), layer=layer
        )

    pad1.connect(
        "e3", r0.ports["e1"], allow_width_mismatch=True, allow_layer_mismatch=True
    )
    pad2.connect(
        "e1", r0.ports["e3"], allow_width_mismatch=True, allow_layer_mismatch=True
    )

    c.info["resistance"] = ohms_per_square * width * length if ohms_per_square else 0
    c.add_port(
        name="pad1",
        port=pad1.ports[pad_port_name],
    )
    c.add_port(
        name="pad2",
        port=pad2.ports[pad_port_name],
    )
    return c


if __name__ == "__main__":
    # import gdsfactory as gf
    # sweep = [resistance_sheet(width=width, layers=((1,0), (1,1))) for width in [1, 10, 100]]
    # c = gf.pack(sweep)[0]

    c = resistance_sheet()
    c.pprint_ports()
    c.show()

    # import gdsfactory as gf
    # sweep_resistance = list(map(resistance_sheet, (5, 10, 80)))
    # c = gf.grid(sweep_resistance)
    # c.show( )
