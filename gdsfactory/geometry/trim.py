"""Trim component.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gdstk

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.component_layout import _parse_layer


@gf.cell
def trim(
    component: Component,
    domain: List[Tuple[float, float]],
    precision: float = 1e-4,
    return_ports: Optional[bool] = False,
) -> Component:
    """Trim a component by another geometry, preserving the component's layers and ports.

    Useful to get a smaller component from a larger one for simulation.

    Args:
        component: Component(/Reference)
        domain: list of array-like[N][2] representing the boundary of the component to keep.
        precision: float Desired precision for rounding vertex coordinates.
        return_ports: whether to return the included ports or not. Ports are always renamed to avoid inheritance conflicts.

    Returns: New component with layers (and possibly ports) of the component restricted to the domain.
    """
    domain_shape = gdstk.Polygon(domain)
    c = Component()

    for layer, layer_polygons in component.get_polygons(by_spec=True).items():
        gds_layer, gds_datatype = _parse_layer(layer)
        for layer_polygon in layer_polygons:
            p = gdstk.boolean(
                operand1=gdstk.Polygon(layer_polygon),
                operand2=domain_shape,
                operation="and",
                precision=precision,
                layer=gds_layer,
                datatype=gds_datatype,
            )
            if p:
                c.add_polygon(p, layer=layer)

    if return_ports:
        ports = []
        i = 0
        for port in component.get_ports():
            if gdstk.inside([port.center], domain_shape):
                new_name = f"{port.name[:1]}{i}"
                ports.append(port.copy(new_name))
                i += 1

        c.add_ports(ports)
        c.auto_rename_ports_layer_orientation()

    return c


if __name__ == "__main__":
    c = gf.components.straight_pin(length=10, taper=None)
    trimmed_c = trim(component=c, domain=[[0, -5], [0, 5], [5, 5], [5, -5]])
    trimmed_c.show(show_ports=True)
