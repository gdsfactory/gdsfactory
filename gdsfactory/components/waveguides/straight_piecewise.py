from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import Section
from gdsfactory.path import Path
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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

    def width_function(_: float) -> npt.NDArray[np.float64]:
        return np.array(widths)

    if isinstance(x, gf.Path):
        p = x
    else:
        p = gf.Path()
        p.points = np.array([(xi, 0.0) for xi in x])

    section_list = list(sections or [])
    section_list.append(
        Section(
            name=name,
            width=0,
            width_function=width_function,
            offset=0,
            layer=layer,
            port_names=port_names,
            **kwargs,
        )
    )
    cross_section = gf.CrossSection(sections=tuple(section_list))

    return gf.path.extrude(p, cross_section=cross_section)


if __name__ == "__main__":
    x, widths = [0, 4.0, 20, 40], [1, 2, 0.15, 0.4]
    c = straight_piecewise(x, widths, "WG")
    c.show()
