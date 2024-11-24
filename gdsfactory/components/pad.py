from __future__ import annotations

import warnings
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass, valid_port_orientations
from gdsfactory.typings import (
    AngleInDegrees,
    ComponentFactory,
    Float2,
    Ints,
    LayerSpec,
    Spacing,
)


@gf.cell
def pad(
    size: str | Float2 = (100.0, 100.0),
    layer: LayerSpec = "MTOP",
    bbox_layers: tuple[LayerSpec, ...] | None = None,
    bbox_offsets: tuple[float, ...] | None = None,
    port_inclusion: float = 0,
    port_orientation: AngleInDegrees | None = 0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    port_type: str = "pad",
) -> Component:
    """Returns rectangular pad with ports.

    Args:
        size: x, y size.
        layer: pad layer.
        bbox_layers: list of layers.
        bbox_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        port_inclusion: from edge.
        port_orientation: in degrees for the center port.
        port_orientations: list of port_orientations to add. None does not add ports.
        port_type: port type for pad port.
    """
    c = Component()
    layer = gf.get_layer(layer)
    size = gf.get_constant(size)
    rect = compass(
        size=size,
        layer=layer,
        port_inclusion=port_inclusion,
        port_type="electrical",
        port_orientations=port_orientations,
    )
    c_ref = c.add_ref(rect)
    c.add_ports(c_ref.ports)
    c.info["size"] = size
    c.info["xsize"] = size[0]
    c.info["ysize"] = size[1]

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

    if port_orientation is not None and port_orientation not in valid_port_orientations:
        raise ValueError(f"{port_orientation=} must be in {valid_port_orientations}")

    width = size[1] if port_orientation in {0, 180} else size[0]

    if port_orientation is not None:
        c.add_port(
            name="pad",
            port_type=port_type,
            layer=layer,
            center=(0, 0),
            orientation=port_orientation,
            width=width,
        )
    c.flatten()
    return c


pad_rectangular = partial(pad, size="pad_size")
pad_small = partial(pad, size=(80, 80))


@gf.cell
def pad_array(
    pad: ComponentFactory = "pad",
    spacing: Spacing | None = None,
    columns: int = 6,
    rows: int = 1,
    column_pitch: float = 150.0,
    row_pitch: float = 150.0,
    port_orientation: AngleInDegrees = 0,
    orientation: AngleInDegrees | None = None,
    size: Float2 = (100.0, 100.0),
    layer: LayerSpec = "MTOP",
    centered_ports: bool = False,
    auto_rename_ports: bool = False,
) -> Component:
    """Returns 2D array of pads.

    Args:
        pad: pad element.
        spacing: x, y pitch.
        columns: number of columns.
        rows: number of rows.
        column_pitch: x pitch.
        row_pitch: y pitch.
        port_orientation: port orientation in deg. None for low speed DC ports.
        orientation: Deprecated, use port_orientation.
        size: pad size.
        layer: pad layer.
        centered_ports: True add ports to center. False add ports to the edge.
        auto_rename_ports: True to auto rename ports.
    """
    if orientation is not None:
        warnings.warn("orientation is deprecated, use port_orientation")
        port_orientation = orientation

    if spacing:
        warnings.warn("spacing is deprecated, use column_pitch and row_pitch")
        column_pitch, row_pitch = spacing

    c = Component()
    pad = gf.get_component(
        pad, size=size, layer=layer, port_orientations=None, port_orientation=None
    )

    c.add_ref(
        pad, columns=columns, rows=rows, column_pitch=column_pitch, row_pitch=row_pitch
    )
    width = size[0] if port_orientation in {90, 270} else size[1]

    for col in range(columns):
        for row in range(rows):
            center = (col * column_pitch, row * row_pitch)
            port_orientation = int(port_orientation)
            center = [center[0], center[1]]

            if not centered_ports:
                if port_orientation == 0:
                    center[0] += size[0] / 2
                elif port_orientation == 90:
                    center[1] += size[1] / 2
                elif port_orientation == 180:
                    center[0] -= size[0] / 2
                elif port_orientation == 270:
                    center[1] -= size[1] / 2

            center = (center[0], center[1])
            c.add_port(
                name=f"e{row + 1}{col + 1}",
                center=center,
                width=width,
                orientation=port_orientation,
                port_type="electrical",
                layer=layer,
            )
    if auto_rename_ports:
        c.auto_rename_ports()
    return c


pad_array90 = partial(pad_array, port_orientation=90)
pad_array270 = partial(pad_array, port_orientation=270)

pad_array0 = partial(pad_array, port_orientation=0, columns=1, rows=3)
pad_array180 = partial(pad_array, port_orientation=180, columns=1, rows=3)


if __name__ == "__main__":
    # c = pad_rectangular()
    # c = pad_array(columns=3, centered_ports=True, port_orientation=90)
    c = pad(port_orientations=[270])
    c.pprint_ports()
    c.show()
