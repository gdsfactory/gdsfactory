from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Protocol

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec, LayerSpecs

_F = Callable[..., Component]

symbol = gf.cell


class ToSymbol(Protocol):
    def __call__(
        self, component: Component, *args: Any, **kwargs: Any
    ) -> Component: ...


def symbol_from_cell(func: _F, to_symbol: ToSymbol) -> _F:
    """Creates a symbol function from a component function.

    Args:
        func: the cell function
        to_symbol: the function that transforms the output of the cell function into a symbol

    Returns:
        a symbol function
    """

    @functools.wraps(func)
    def _symbol(*args: Any, **kwargs: Any) -> Component:
        component = func(*args, **kwargs)
        c_symbol = to_symbol(component, prefix=f"SYMBOL_{func.__name__}")
        return c_symbol

    _symbol._symbol = True  # type: ignore[attr-defined]
    return _symbol


@symbol
def floorplan_with_block_letters(
    component: Component,
    copy_layers: LayerSpecs = ("WG",),
    text_layer: LayerSpec = (2, 0),
    bbox_layer: LayerSpec = (90, 0),
) -> Component:
    """Returns symbol.

    Keeps same floorplan as component layout, function name \
        and optionally shapes on layers copied from the original layout.

    Args:
        component: the layout component.
        copy_layers: if specified, copies layers from the layout into the symbol.
        text_layer: the layer for the text.
        bbox_layer: the layer for the bounding box.

    Returns:
        A component representing the symbol.

    """
    import gdsfactory as gf
    from gdsfactory.components import rectangle, text

    w = component.dsize_info.width
    h = component.dsize_info.height
    sym = Component()

    # add floorplan box
    bbox = sym << rectangle(size=(w, h), layer=bbox_layer)
    bbox.x = component.x
    bbox.y = component.y

    # add text, fit to box with specified margin
    margin = 0.2
    max_w, max_h = w * (1 - margin), h * (1 - margin)
    text_init_size = 3.0
    text_init = text(
        component.function_name or "",
        size=text_init_size,
        layer=text_layer,
        justify="center",
    )
    w_text = text_init.dsize_info.width
    h_text = text_init.dsize_info.height

    w_scaling = max_w / w_text
    h_scaling = max_h / h_text
    scaling = min(w_scaling, h_scaling)
    text_size = text_init_size * scaling
    text_component = text(
        component.function_name or "",
        size=text_size,
        layer=text_layer,
        justify="center",
    )

    text = sym << text_component
    text.x = component.x
    text.y = component.y

    sym.add_ports(component.ports)

    # add specified layers from original layout
    if copy_layers:
        for layer in copy_layers:
            layer = gf.get_layer(layer)
            polys = component.get_polygons(merge=True)[layer]
            for poly in polys:
                sym.add_polygon(poly, layer=layer)

    return sym


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.c.mmi1x2()
    s = floorplan_with_block_letters(c)
    s.show()
