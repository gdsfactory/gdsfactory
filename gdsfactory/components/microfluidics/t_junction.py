from __future__ import annotations

__all__ = ["t_junction"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["microfluidics"])
def t_junction(
    main_width: float = 1.0,
    branch_width: float = 1.0,
    main_length: float = 20.0,
    branch_length: float = 10.0,
    reservoir_radius: float = 0.0,
    n_reservoir_points: int = 64,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a microfluidic T-junction.

    A horizontal main channel with a vertical branch going up from
    the center. Optionally adds circular reservoirs at the three
    endpoints.

    Args:
        main_width: width of the main horizontal channel.
        branch_width: width of the vertical branch channel.
        main_length: total length of the main channel.
        branch_length: length of the vertical branch.
        reservoir_radius: radius of circular reservoirs at endpoints (0 to disable).
        n_reservoir_points: number of polygon points for reservoir circles.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.t_junction(
        main_width=main_width,
        branch_width=branch_width,
        main_length=main_length,
        branch_length=branch_length,
        reservoir_radius=reservoir_radius,
        n_reservoir_points=n_reservoir_points,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = t_junction()
    c.show()
