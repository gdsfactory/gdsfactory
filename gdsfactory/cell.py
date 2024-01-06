"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import warnings

from kfactory import cell

import gdsfactory as gf

cell_without_validator = cell


def clear_cache() -> None:
    """Clear the cache of the cell decorator."""
    warnings.warn("clear_cache is deprecated and does nothing in gdsfactory>=8.0.0")


@cell
def container(component, function, **kwargs) -> gf.Component:
    """Returns new component with a component reference.

    Args:
        component: to add to container.
        function: function to apply to component.
        kwargs: keyword arguments to pass to function.

    """

    component = gf.get_component(component)
    c = gf.Component()
    cref = c << component
    c.add_ports(cref.ports)
    function(c, **kwargs)
    c.copy_child_info(component)
    return c


__all__ = ["cell", "cell_without_validator", "container"]
