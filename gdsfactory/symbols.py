from __future__ import annotations

import functools
from collections.abc import Callable

from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpecs

_F = Callable[..., Component]

symbol = cell


def symbol_from_cell(func: _F, to_symbol: Callable[[Component, ...], Component]) -> _F:
    """Creates a symbol function from a component function.

    Args:
        func: the cell function
        to_symbol: the function that transforms the output of the cell function into a symbol

    Returns:
        a symbol function
    """

    @functools.wraps(func)
    def _symbol(*args, **kwargs):
        component = func(*args, **kwargs)
        symbol = to_symbol(component, prefix=f"SYMBOL_{func.__name__}")
        return symbol

    _symbol._symbol = True
    return _symbol


@symbol
def floorplan_with_block_letters(
    component: Component, copy_layers: LayerSpecs = ("WG",)
) -> Component:
    """Returns symbol.

    Keeps same floorplan as component layout, function name \
        and optionally shapes on layers copied from the original layout.

    Args:
        component: the layout component.
        copy_layers: if specified, copies layers from the layout into the symbol.

    Returns:
        A component representing the symbol.

    """
    import gdsfactory as gf
    from gdsfactory.components import rectangle, text

    w = component.dsize_info.width
    h = component.dsize_info.height
    sym = Component()

    # add floorplan box
    bbox = sym << rectangle(size=(w, h), layer=(0, 0))
    bbox.dmove((0, 0), other=component.bbox[0])

    # add text, fit to box with specified margin
    margin = 0.2
    max_w, max_h = w * (1 - margin), h * (1 - margin)
    text_init_size = 3.0
    text_init = text(
        component.settings.function_name,
        size=text_init_size,
        layer=(2, 0),
        justify="center",
    )
    w_text = text_init.dsize_info.width
    h_text = text_init.dsize_info.height

    w_scaling = max_w / w_text
    h_scaling = max_h / h_text
    scaling = min(w_scaling, h_scaling)
    text_size = text_init_size * scaling
    text_component = text(
        component.settings.function_name, size=text_size, layer=(2, 0), justify="center"
    )

    text = sym << text_component
    text.dmove(text.dcenter, other=bbox.dcenter)

    # add ports
    sym.add_ports(component.ports)

    # add specified layers from original layout
    if copy_layers:
        for layer in copy_layers:
            layer = gf.get_layer(layer)
            polys = component.get_polygons(merge=True)[layer]
            sym.add_polygon(polys, layer=layer)

    return sym
