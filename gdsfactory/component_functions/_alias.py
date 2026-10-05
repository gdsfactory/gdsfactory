"""Cell aliases: preset settings for a cell, looked up by name when called."""

from __future__ import annotations

from functools import partial
from typing import Any

from gdsfactory.component import Component


class CellAlias(partial[Component]):
    """A cell with preset settings that looks up its base cell by name.

    It behaves like ``functools.partial(cell, **settings)``, so it serializes,
    names its cells and shows its signature the same way. When called, it gets
    the cell named ``cell.__name__`` from the active PDK instead of calling
    ``cell`` directly, so a PDK override of the base cell also applies to its
    aliases. An alias of an alias is flattened into one alias of the base cell.

    ```python
    via1 = CellAlias(via, layer="VIA1")
    ```
    """

    def __new__(cls, func: Any, /, *args: Any, **keywords: Any) -> CellAlias:
        if args:
            raise TypeError("CellAlias takes keyword settings only.")
        if isinstance(func, partial):
            if func.args:
                raise TypeError("CellAlias takes keyword settings only.")
            keywords = {**func.keywords, **keywords}
            func = func.func
        return super().__new__(cls, func, **keywords)

    def __call__(self, /, *args: Any, **kwargs: Any) -> Component:
        from gdsfactory.component_functions._get_component import get_component

        if args:
            raise TypeError(f"{self!r} takes keyword arguments only.")
        return get_component(self.func.__name__, settings={**self.keywords, **kwargs})
