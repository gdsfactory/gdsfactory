from __future__ import annotations

__all__ = ["pad", "pad_array"]

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.config import valid_port_orientations
from gdsfactory.typings import (
    AngleInDegrees,
    ComponentSpec,
    CrossSectionSpec,
    Float2,
    Ints,
    LayerSpec,
    Size,
)


def pad(
    size: Size | str = (100.0, 100.0),
    layer: LayerSpec | None = None,
    bbox_layers: tuple[LayerSpec, ...] | None = None,
    bbox_offsets: tuple[float, ...] | None = None,
    port_inclusion: float = 0,
    port_orientation: AngleInDegrees | None = 0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    port_type: str = "pad",
    cross_section: CrossSectionSpec | None = "metal_routing",
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
        cross_section: profile of the pad ports, resized to each port's width.
            Also gives the default layer. Not used for the ports if `layer` is a
            different layer, as no profile of this cross section fits the pad.
    """
    c = Component()
    if cross_section is not None:
        xs_layer = gf.get_layer(gf.get_cross_section(cross_section).layer)
        if layer is None:
            layer = xs_layer
        elif gf.get_layer(layer) != xs_layer:
            cross_section = None
    if layer is None:
        raise ValueError("pad needs a layer or a cross_section")
    layer = gf.get_layer(layer)
    size_ = gf.get_constant(size)
    rect = get_component(
        "compass",
        size=size_,
        layer=layer,
        port_inclusion=port_inclusion,
        port_type="electrical",
        port_orientations=port_orientations,
        **({"cross_section": cross_section} if cross_section is not None else {}),
    )
    c_ref = c.add_ref(rect)
    c.add_ports(c_ref.ports)
    c.info["size"] = size_
    c.info["xsize"] = size_[0]
    c.info["ysize"] = size_[1]

    if port_orientation is not None and port_orientation not in valid_port_orientations:
        raise ValueError(f"{port_orientation=} must be in {valid_port_orientations}")

    width = size_[1] if port_orientation in {0, 180} else size_[0]

    if port_orientation is not None:
        profile: dict[str, Any] = (
            {"width": width, "layer": layer}
            if cross_section is None
            else {"cross_section": gf.get_cross_section(cross_section, width=width)}
        )
        c.add_port(
            name="pad",
            port_type=port_type,
            center=(0, 0),
            orientation=port_orientation,
            **profile,
        )

    if bbox_layers and bbox_offsets:
        sizes: list[Size] = []
        for cladding_offset in bbox_offsets:
            size_new = (size_[0] + 2 * cladding_offset, size_[1] + 2 * cladding_offset)
            sizes.append(size_new)

        for layer, size_new in zip(bbox_layers, sizes, strict=False):
            c.add_ref(
                get_component(
                    "compass",
                    size=size_new,
                    layer=layer,
                )
            )
    c.flatten()
    elec = [p for p in c.ports if p.port_type in {"electrical", "pad"}]
    if elec:
        c.create_pin(ports=elec, name="pad")
    return c


def pad_array(
    pad: ComponentSpec = "pad",
    columns: int = 6,
    rows: int = 1,
    column_pitch: float = 150.0,
    row_pitch: float = 150.0,
    port_orientation: AngleInDegrees = 0,
    size: Float2 | None = None,
    layer: LayerSpec | None = "MTOP",
    centered_ports: bool = False,
    auto_rename_ports: bool = False,
) -> Component:
    """Returns 2D array of pads.

    Args:
        pad: pad element.
        columns: number of columns.
        rows: number of rows.
        column_pitch: x pitch.
        row_pitch: y pitch.
        port_orientation: port orientation in deg. None for low speed DC ports.
        size: pad size.
        layer: pad layer.
        centered_ports: True add ports to center. False add ports to the edge.
        auto_rename_ports: True to auto rename ports.
    """
    c = Component()

    pad_kwargs: dict[str, Any] = {}
    if layer is not None:
        pad_kwargs["layer"] = layer
    if size is not None:
        pad_kwargs["size"] = size
    pad_component = get_component(
        pad,
        port_orientation=port_orientation,
        port_orientations=(port_orientation,) if not centered_ports else None,
        **pad_kwargs,
    )

    pad_size: Float2 = size or pad_component.info["size"]
    pad_layer: LayerSpec = layer or pad_component.ports[0].layer

    c.add_ref(
        pad_component,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )
    width = pad_size[0] if int(port_orientation) in {90, 270} else pad_size[1]
    pad_port = next((p for p in pad_component.ports if p.name == "pad"), None)
    profile: dict[str, Any] = (
        {"cross_section": pad_port.cross_section}
        if pad_port is not None
        and pad_port.width == width
        and gf.get_layer(pad_port.layer) == gf.get_layer(pad_layer)
        else {"width": width, "layer": pad_layer}
    )

    for col in range(columns):
        for row in range(rows):
            center = (col * column_pitch, row * row_pitch)
            port_orientation = int(port_orientation)
            center_list = [center[0], center[1]]

            if not centered_ports:
                if port_orientation == 0:
                    center_list[0] += pad_size[0] / 2
                elif port_orientation == 90:
                    center_list[1] += pad_size[1] / 2
                elif port_orientation == 180:
                    center_list[0] -= pad_size[0] / 2
                elif port_orientation == 270:
                    center_list[1] -= pad_size[1] / 2

            center = (center_list[0], center_list[1])
            c.add_port(
                name=f"e{row + 1}{col + 1}",
                center=center,
                orientation=port_orientation,
                port_type="electrical",
                **profile,
            )
    if auto_rename_ports:
        c.auto_rename_ports()
    for port in c.ports:
        if port.port_type == "electrical":
            c.create_pin(ports=[port], name=f"pad_{port.name}")
    return c
