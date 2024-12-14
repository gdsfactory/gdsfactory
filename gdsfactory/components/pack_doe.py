from __future__ import annotations

import itertools as it
from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.grid import grid, grid_with_text
from gdsfactory.pack import pack
from gdsfactory.typings import CellSpec, ComponentSpec


def generate_doe(
    doe: ComponentSpec,
    settings: dict[str, Sequence[Any]],
    do_permutations: bool = False,
    function: CellSpec | None = None,
) -> tuple[list[Component], list[dict[str, Any]]]:
    """Generates a component DOE (Design of Experiment).

    which can then be packed, or used elsewhere.

    Args:
        doe: function to return Components.
        settings: component settings.
        do_permutations: for each setting.
        function: for the component (add padding, grating couplers ...)
    """
    if do_permutations:
        settings_list = [dict(zip(settings, t)) for t in it.product(*settings.values())]
    else:
        settings_list = [dict(zip(settings, t)) for t in zip(*settings.values())]

    if function:
        function = gf.get_cell(function)
        if not callable(function):
            raise ValueError(f"Error {function!r} needs to be callable.")
        component_list = [
            function(gf.get_component(doe, **settings)) for settings in settings_list
        ]
    else:
        component_list = [
            gf.get_component(doe, **settings) for settings in settings_list
        ]

    return component_list, settings_list


def pack_doe(
    doe: ComponentSpec,
    settings: dict[str, Sequence[Any]],
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
    component_list, settings_list = generate_doe(
        doe=doe, settings=settings, do_permutations=do_permutations, function=function
    )

    components = pack(component_list, **kwargs)

    if len(components) > 1:
        raise ValueError(
            f"failed to pack in one Component, it created {len(components)} Components"
        )
    component = components[0]
    component.doe_names = [component.name for component in component_list]  # type: ignore
    component.doe_settings = settings_list  # type: ignore
    return component


def pack_doe_grid(
    doe: ComponentSpec,
    settings: dict[str, Sequence[Any]],
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
    if do_permutations:
        settings_list = [dict(zip(settings, t)) for t in it.product(*settings.values())]
    else:
        settings_list = [dict(zip(settings, t)) for t in zip(*settings.values())]

    if function:
        function = gf.get_cell(function)
        if not callable(function):
            raise ValueError(f"Error {function!r} needs to be callable.")
        component_list = [
            function(gf.get_component(doe, **settings)) for settings in settings_list
        ]
    else:
        component_list = [
            gf.get_component(doe, **settings) for settings in settings_list
        ]

    if with_text:
        c = grid_with_text(component_list, **kwargs)

    else:
        c = grid(component_list, **kwargs)

    c.doe_names = [component.name for component in component_list]  # type: ignore
    c.doe_settings = settings_list  # type: ignore
    return c


if __name__ == "__main__":
    c = pack_doe_grid(
        doe="mmi1x2",
        settings=dict(length_mmi=(2.5, 100), width_mmi=(4, 10)),
        with_text=True,
        spacing=(100, 100),
        shape=(2, 2),
        do_permutations=True,
    )

    c = pack_doe(doe="mmi1x2", settings=dict(length_mmi=(2, 100), width_mmi=(4, 10)))
    # c = pack_doe()
    c.show()
