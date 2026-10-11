from __future__ import annotations

__all__ = ["add_frame", "align_wafer"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell_with_module_name(tags=["dies"])
def align_wafer(
    width: float = 10.0,
    spacing: float = 10.0,
    cross_length: float = 80.0,
    layer: LayerSpec = "WG",
    layer_cladding: tuple[int, int] | None = None,
    square_corner: str = "bottom_left",
) -> Component:
    """Returns cross inside a frame to align wafer.

    Args:
        width: in um.
        spacing: in um.
        cross_length: for the cross.
        layer: for the cross.
        layer_cladding: optional.
        square_corner: bottom_left, bottom_right, top_right, top_left.
    """
    return cf.align_wafer(
        width=width,
        spacing=spacing,
        cross_length=cross_length,
        layer=layer,
        layer_cladding=layer_cladding,
        square_corner=square_corner,
    )


@gf.cell_with_module_name(tags=["dies"])
def add_frame(
    component: ComponentSpec = "rectangle",
    width: float = 10.0,
    spacing: float = 10.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns component with a frame around it.

    Args:
        component: Component to frame.
        width: of the frame.
        spacing: of component to frame.
        layer: frame layer.
    """
    return cf.add_frame(
        component=component,
        width=width,
        spacing=spacing,
        layer=layer,
    )
