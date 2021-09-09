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
) -> Component:
    """Rectangular pad with 4 ports (1, 2, 3, 4)

    Args:
        width: pad width
        height: pad height
        layer: pad layer
        layers_cladding:
        cladding_offsets:
    """
    c = Component()
    rect = compass(size=size, layer=layer)
    c_ref = c.add_ref(rect)
    c.add_ports(c_ref.ports)
    c.absorb(c_ref)

    if layers_cladding and cladding_offsets:
        for layer, cladding_offset in zip(layers_cladding, cladding_offsets):
            c.add_ref(
                compass(
                    size=(size[0] + 2 * cladding_offset, size[1] + 2 * cladding_offset),
                    layer=layer,
                )
            )

    return c


@cell
def pad_array(
    pad: ComponentOrFactory = pad,
    pitch: float = 150.0,
    n: int = 6,
    port_names: Tuple[str, ...] = ("e4",),
    axis: str = "x",
) -> Component:
    """Returns 1D array of rectangular pads

    Args:
        pad: pad element
        pitch: x spacing
        n: number of pads
        port_names: per pad (e1: west, e2: north, e3: east, e4: south)
        pad_settings: settings for pad if pad is callable
        axis: x or y
    """
    c = Component()
    pad = pad() if callable(pad) else pad
    port_names = list(port_names)

    for i in range(n):
        p = c << pad
        if axis == "x":
            p.x = i * pitch
        elif axis == "y":
            p.y = i * pitch
        else:
            raise ValueError(f"Invalid axis {axis} not in (x, y)")
        for port_name in port_names:
            port_name_new = f"e{i+1}"
            c.add_port(name=port_name_new, port=p.ports[port_name])
    return c


pad_array180 = partial(pad_array, port_names=("e1",))
pad_array90 = partial(pad_array, port_names=("e2",))
pad_array0 = partial(pad_array, port_names=("e3",))
pad_array270 = partial(pad_array, port_names=("e4",))


@cell
def pad_array_2d(
    pad: ComponentOrFactory = pad,
    pitch_x: float = 150.0,
    pitch_y: float = 150.0,
    cols: int = 3,
    rows: int = 3,
    port_names: Tuple[str, ...] = ("e2",),
    **kwargs,
) -> Component:
    """Returns 2D array of rectangular pads

    the ports names are e{row}_{col}

    Args:
        pad: pad element
        pitch_x: horizontal x spacing
        pitch_y: vertical y spacing
        cols: number of cols
        rows: number of rows
        port_names: list of port names (N, S, W, E) per pad
        **kwargs: settings for pad if pad is callable
    """
    c = Component()
    pad = pad(**kwargs) if callable(pad) else pad
    port_names = list(port_names)

    for j in range(rows):
        for i in range(cols):
            p = c << pad
            p.x = i * pitch_x
            p.y = j * pitch_y
            for port_name in port_names:
                if port_name not in p.ports:
                    raise ValueError(f"{port_name} not in {list(p.ports.keys())}")
                port_name_new = f"e{j+1}_{i+1}"
                c.add_port(port=p.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":
    # c = pad()

    # c = pad(layer_to_inclusion={(3, 0): 10})
    # print(c.ports)
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # print(c.settings['spacing'])
    # c = pad_array90()
    # c = pad_array270()
    # c.pprint_ports
    c = pad_array_2d(cols=2, rows=3, port_names=("e2",))
    c.show()
