from __future__ import annotations

__all__ = ["arrow_junction"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["microfluidics"])
def arrow_junction(
    main_width: float = 1.0,
    branch_width: float = 1.0,
    main_length: float = 20.0,
    branch_length: float = 10.0,
    branch_angle: float = 35.0,
    reservoir_radius: float = 0.0,
    n_reservoir_points: int = 64,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a microfluidic arrow (Y) junction.

    A main horizontal channel on the right side with two angled branches
    diverging to the left at +/- branch_angle from horizontal. The
    junction point is at the origin.

    Args:
        main_width: width of the main horizontal channel.
        branch_width: width of each angled branch channel.
        main_length: length of the main channel (extends to the right).
        branch_length: length of each angled branch.
        branch_angle: angle of each branch from horizontal (degrees).
        reservoir_radius: radius of circular reservoirs at endpoints (0 to disable).
        n_reservoir_points: number of polygon points for reservoir circles.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    return cf.arrow_junction(
        main_width=main_width,
        branch_width=branch_width,
        main_length=main_length,
        branch_length=branch_length,
        branch_angle=branch_angle,
        reservoir_radius=reservoir_radius,
        n_reservoir_points=n_reservoir_points,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = arrow_junction()
    c.show()
