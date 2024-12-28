from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import Section
from gdsfactory.path import Path
from gdsfactory.typings import LayerSpec


@gf.cell
def straight_piecewise(
    x: Sequence[float] | Path,
    y: Sequence[float],
    layer: LayerSpec,
    sections: Sequence[Section] | None = None,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    name: str = "core",
    **kwargs: Any,
) -> Component:
    """Create a component with a piecewise waveguide.

    Args:
        x (Union[Sequence[float], gf.Path]): x coordinates of the piecewise function or a custom path.
        y (Sequence[float]): y coordinates of the piecewise function.
        layer (LayerSpec): layer to extrude.
        sections (Sequence[Section] | None): sections to extrude.
        port_names (tuple[str | None, str | None]): port names.
        name (str, optional): name of the component.
        **kwargs: additional keyword arguments for the Section.
    """
    if isinstance(x, Sequence) and len(x) != len(y):
        raise ValueError("x and y must have the same length")

    def width_function(_: float) -> npt.NDArray[np.floating[Any]]:
        return np.array(y)

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
    x, y = [0, 4.0, 20, 40], [1, 2, 0.15, 0.4]
    c = straight_piecewise(x, y, "WG")
    c.show()
