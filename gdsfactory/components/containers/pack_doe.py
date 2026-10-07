from __future__ import annotations

__all__ = ["generate_doe", "pack_doe", "pack_doe_grid"]

from collections.abc import Mapping, Sequence
from typing import Any

import kfactory as kf

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.containers.pack_doe import generate_doe
from gdsfactory.typings import CellSpec, ComponentSpec


@gf.cell_with_module_name(tags=["containers"])
def pack_doe(
    doe: ComponentSpec | None = None,
    settings: Mapping[str, Sequence[kf.typings.MetaData]] | None = None,
    do_permutations: bool = False,
    function: CellSpec | None = None,
    **kwargs: Any,
) -> Component:
    """Packs a component DOE (Design of Experiment) using pack.

    Args:
        doe: function to return Components.
        settings: component settings.
        do_permutations: for each setting.
        function: to apply (add padding, grating couplers).
        kwargs: for pack.

    Keyword Args:
        spacing: Minimum distance between adjacent shapes.
        aspect_ratio: (width, height) ratio of the rectangular bin.
        max_size: Limits the size into which the shapes will be packed.
        sort_by_area: Pre-sorts the shapes by area.
        density: Values closer to 1 pack tighter but require more computation.
        precision: Desired precision for rounding vertex coordinates.
        text: Optional function to add text labels.
        text_prefix: for labels. For example. 'A' for 'A1', 'A2'...
        text_offsets: relative to component size info anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        name_prefix: for each packed component (avoids the Unnamed cells warning).
            Note that the suffix contains a uuid so the name will not be deterministic.
        rotation: for each component in degrees.
        h_mirror: horizontal mirror in y axis (x, 1) (1, 0). This is the most common.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
    """
    return cf.pack_doe(
        doe=doe,
        settings=settings,
        do_permutations=do_permutations,
        function=function,
        **kwargs,
    )


@gf.cell_with_module_name(tags=["containers"])
def pack_doe_grid(
    doe: ComponentSpec | None = None,
    settings: Mapping[str, Sequence[kf.typings.MetaData]] | None = None,
    do_permutations: bool = False,
    function: CellSpec | None = None,
    with_text: bool = False,
    **kwargs: Any,
) -> Component:
    """Packs a component DOE (Design of Experiment) using grid.

    Args:
        doe: function to return Components.
        settings: component settings.
        do_permutations: for each setting.
        function: to apply to component (add padding, grating couplers).
        with_text: includes text label.
        kwargs: for grid.

    Keyword Args:
        spacing: between adjacent elements on the grid, can be a tuple for
        different distances in height and width.
        separation: If True, guarantees elements are separated with fixed spacing
        if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape).
        If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'} for x (column) alignment along.
        align_y: {'y', 'ymin', 'ymax'} for y (row) alignment along.
        edge_x: {'x', 'xmin', 'xmax'} for x (column) (ignored if separation = True).
        edge_y: {'y', 'ymin', 'ymax'} for y (row) (ignored if separation = True).
        rotation: for each component in degrees.
        h_mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
    """
    return cf.pack_doe_grid(
        doe=doe,
        settings=settings,
        do_permutations=do_permutations,
        function=function,
        with_text=with_text,
        **kwargs,
    )


if __name__ == "__main__":
    c = pack_doe_grid()
    c.show()
