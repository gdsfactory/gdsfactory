from __future__ import annotations

import functools
from typing import Callable

import gdstk
from pydantic import validate_arguments

from gdsfactory.cell import _F, cell_without_validator
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpecs


def symbol(func: _F, *args, **kwargs) -> _F:
    """Decorator for Component symbols.

    Wraps cell_without_validator
    Validates type annotations with pydantic.

    Implements a cache so that if a symbol has already been built
    it will return the component from the cache directly.
    This avoids 2 exact cells that are not references of the same cell
    You can always over-ride this with `cache = False`.

    When decorate your functions with @cell you get:

    - CACHE: avoids creating duplicated cells.
    - name: gives Components a unique name based on parameters.
    - adds Component.info with default, changed and full component settings.

    Keyword Args:
        autoname (bool): if True renames component based on args and kwargs.
        name (str): Optional (ignored when autoname=True).
        cache (bool): returns component from the cache if it already exists.
            if False creates a new component.
            by default True avoids having duplicated cells with the same name.
        info: updates component.info dict.
        prefix: name_prefix, defaults to function name.
        max_name_length: truncates name beyond some characters (32) with a hash.
        decorator: function to run over the component.
    """
    if "prefix" not in kwargs:
        prefix = f"SYMBOL_{func.__name__}"
        kwargs["prefix"] = prefix
    _wrapped = functools.partial(
        cell_without_validator(validate_arguments(func)), **kwargs
    )
    _wrapped._symbol = True
    return _wrapped


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
    """Returns symbol with same floorplan as component layout, function name \
        and optionally shapes on layers copied from the original layout.

    Args:
        component: the layout component.
        copy_layers: if specified, copies layers from the layout into the symbol.

    Returns:
        A component representing the symbol.

    """
    import gdsfactory as gf
    from gdsfactory.components import rectangle, text

    w, h = component.size
    sym = Component()

    # add floorplan box
    bbox = sym << rectangle(size=(w, h), layer=(0, 0))
    bbox.move((0, 0), destination=component.bbox[0])

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
    w_text, h_text = text_init.size
    w_scaling = max_w / w_text
    h_scaling = max_h / h_text
    scaling = min(w_scaling, h_scaling)
    text_size = text_init_size * scaling
    text_component = text(
        component.settings.function_name, size=text_size, layer=(2, 0), justify="center"
    )

    text = sym << text_component
    text.move(text.center, destination=bbox.center)

    # add ports
    sym.add_ports(component.ports)

    # add specified layers from original layout
    if copy_layers:
        for layer in copy_layers:
            layer = gf.get_layer(layer)
            polys = component.get_polygons(by_spec=layer, as_array=False)
            # run OR to simplify shapes
            polys = gdstk.boolean(polys, [], "or", layer=layer[0], datatype=layer[1])
            sym.add_polygon(polys)

    return sym
