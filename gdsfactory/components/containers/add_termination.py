from __future__ import annotations

__all__ = ["add_termination", "taper_terminator"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec

from ..tapers.taper import taper

taper_terminator = CellAlias(taper, width2=0.1)


@gf.cell_with_module_name(tags=["containers"])
def add_termination(
    component: ComponentSpec = "straight",
    port_names: tuple[str, ...] | None = None,
    terminator: ComponentSpec = "taper_terminator",
    terminator_port_name: str | None = None,
) -> Component:
    """Returns component with terminator on some ports.

    Args:
        component: to add terminator.
        port_names: ports to add terminator.
        terminator: factory for the terminator.
        terminator_port_name: for the terminator to connect to the component ports.
    """
    return cf.add_termination(
        component=component,
        port_names=port_names,
        terminator=terminator,
        terminator_port_name=terminator_port_name,
    )
