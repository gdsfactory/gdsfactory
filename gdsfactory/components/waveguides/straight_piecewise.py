__all__ = ["straight_piecewise"]

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
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
    return cf.straight_piecewise(
        x=x,
        widths=widths,
        layer=layer,
        sections=sections,
        port_names=port_names,
        name=name,
        **kwargs,
    )
