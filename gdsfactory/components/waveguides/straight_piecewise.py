__all__ = ["straight_piecewise"]

from collections.abc import Sequence
from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import Section
from gdsfactory.path import Path
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["waveguides"])
def straight_piecewise(
    x: Sequence[float] | Path,
    widths: Sequence[float],
    layer: LayerSpec,
    sections: Sequence[Section] | None = None,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    name: str = "core",
    **kwargs: Any,
) -> Component:
    """Create a component with a piecewise-defined straight waveguide.

    Args:
        x: X coordinates or a custom Path object.
        widths: Waveguide widths at each corresponding x.
        layer: Layer to extrude.
        sections: Additional cross-section sections to extrude.
        port_names: Port names for the waveguide.
        name: Name for the core (main) Section.
        **kwargs: Additional keyword arguments for the Section.
    """
    if isinstance(x, Sequence) and len(x) != len(widths):
        raise ValueError("x and widths must have the same length.")

    if isinstance(x, gf.Path):
        p = x
    else:
        p = gf.Path()
        p.points = np.array([(xi, 0.0) for xi in x])

    section_list = list(sections or [])
    if not widths:
        raise ValueError("widths must contain at least one value")

    xs1 = gf.cross_section.cross_section(
        width=float(widths[0]),
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )
    xs2 = gf.cross_section.cross_section(
        width=float(widths[-1]),
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )
    width_values = np.asarray(widths, dtype=float)
    width_positions = np.linspace(0.0, 1.0, len(width_values))
    transition = gf.path.transition(
        xs1,
        xs2,
        width_type="linear",
        core_width_profile=lambda t: np.interp(t, width_positions, width_values),
    )
    return gf.path.extrude_transition(
        p,
        transition=transition,
        port_names=port_names,
    )
