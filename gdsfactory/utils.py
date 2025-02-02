from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

import kfactory as kf
import klayout.db as kdb

if TYPE_CHECKING:
    from gdsfactory.component import ComponentSpec
    from gdsfactory.typings import BoundingBox, Coordinate


def to_kdb_dboxes(
    bounding_boxes: "Sequence[BoundingBox | kdb.DBox | kdb.Box]",
) -> list[kdb.DBox]:
    return [
        box
        if isinstance(box, kdb.DBox)
        else box.to_dtype()
        if isinstance(box, kdb.Box)
        else kdb.DBox(*map(int, box))
        for box in bounding_boxes
    ]


def to_kdb_dpoints(
    points: "Sequence[Coordinate | kdb.Point | kdb.DPoint]",
) -> list[kdb.DPoint]:
    return [
        point
        if isinstance(point, kdb.DPoint)
        else (
            kdb.DPoint(point[0], point[1])
            if isinstance(point, tuple)
            else kdb.DPoint(point.x, point.y)
        )
        for point in points
    ]


def is_component_spec(obj: Any) -> "TypeGuard[ComponentSpec]":
    return isinstance(obj, str | Callable | dict | kf.DKCell)
