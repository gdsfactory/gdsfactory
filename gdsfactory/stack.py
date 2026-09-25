"""Incremental stacking utility — place refs into a parent component one at a time."""

from __future__ import annotations

from typing import Literal

from gdsfactory.component import Component, ComponentReference


def stack_add(
    parent: Component,
    refs: list[ComponentReference],
    child: Component,
    direction: Literal["vertical", "horizontal"] = "vertical",
    spacing: float = 0.0,
    align: Literal["min", "max", "center"] | None = None,
    rotate: float | None = None,
    rows: int = 1,
    columns: int = 1,
    row_pitch: float = 0.0,
    column_pitch: float = 0.0,
) -> ComponentReference:
    """Add a component reference to parent, positioned relative to the last ref in refs.

    Args:
        parent: Component to add the reference into.
        refs: Mutable history list; the last entry drives positioning. Appended in-place.
        child: Component to reference.
        direction: Stack axis — "vertical" (y) or "horizontal" (x).
        spacing: Gap between refs in um. Negative values produce overlap.
        align: Align on the orthogonal axis — "min", "max", or "center". None = no-op.
        rotate: Rotate ref by this angle (degrees) before positioning.
        rows: Array rows passed to add_ref.
        columns: Array columns passed to add_ref.
        row_pitch: Row-to-row pitch (um) passed to add_ref.
        column_pitch: Column-to-column pitch (um) passed to add_ref.

    Returns:
        The newly added ComponentReference.

    Example::

        import gdsfactory as gf
        from gdsfactory.stack import stack_add

        parent = gf.Component()
        refs = []
        stack_add(parent, refs, gf.components.rectangle(size=(10, 5)))
        stack_add(parent, refs, gf.components.rectangle(size=(10, 5)), spacing=2.0)
    """
    ref = parent.add_ref(
        child,
        rows=rows,
        columns=columns,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
    )

    if rotate is not None:
        ref.rotate(rotate)

    if refs:
        prev = refs[-1]
        if direction == "vertical":
            ref.ymax = prev.ymin - spacing
        else:
            ref.xmin = prev.xmax + spacing

        if align is not None:
            if direction == "vertical":
                if align == "min":
                    ref.xmin = prev.xmin
                elif align == "max":
                    ref.xmax = prev.xmax
                else:
                    ref.x = prev.x
            else:  # horizontal
                if align == "min":
                    ref.ymin = prev.ymin
                elif align == "max":
                    ref.ymax = prev.ymax
                else:
                    ref.y = prev.y

    refs.append(ref)
    return ref
