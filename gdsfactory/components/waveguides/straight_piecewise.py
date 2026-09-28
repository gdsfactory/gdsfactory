__all__ = ["straight_piecewise"]

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import SectionSpec
from gdsfactory.path import Path
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["waveguides"])
def straight_piecewise(
    x: Sequence[float] | Path,
    widths: Sequence[float],
    layer: LayerSpec,
    sections: Sequence[SectionSpec] | None = None,
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
        name: Name for the core (main) SectionSpec.
        **kwargs: Additional keyword arguments for the SectionSpec.
    """
    if isinstance(x, Sequence) and len(x) != len(widths):
        raise ValueError("x and widths must have the same length.")

    def width_function(_: float) -> npt.NDArray[np.float64]:
        return np.array(widths)

    if isinstance(x, gf.Path):
        p = x
    else:
        p = gf.Path()
        p.points = np.array([(xi, 0.0) for xi in x])

    cross_section = gf.cross_section.cross_section(
        width=widths[0], layer=layer, sections=sections
    )
    return gf.path.extrude(
        p,
        cross_section=cross_section,
        width_function=width_function,
        ports={0: (*port_names, "optical")},
        **kwargs,
    )
