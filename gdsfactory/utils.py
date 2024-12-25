from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

import kfactory as kf
import klayout.db as kdb

if TYPE_CHECKING:
    from gdsfactory.typings import BoundingBox, ComponentSpec, Coordinate


def to_kdb_boxes(bounding_boxes: "Sequence[BoundingBox | kdb.Box]") -> list[kdb.Box]:
    return [
        box if isinstance(box, kdb.Box) else kdb.Box(*map(int, box))
        for box in bounding_boxes
    ]


def to_kdb_dpoints(
    points: "Sequence[Coordinate | kdb.Point | kdb.DPoint]",
) -> list[kdb.DPoint]:
    return [
        point if isinstance(point, kdb.DPoint) else kdb.DPoint(*point)
        for point in points
    ]


def to_kdb_points(
    points: "Sequence[Coordinate | kdb.Point | kdb.DPoint]",
) -> list[kdb.Point]:
    return [
        point if isinstance(point, kdb.Point) else kdb.Point(*point) for point in points
    ]


def is_component_spec(obj: Any) -> "TypeGuard[ComponentSpec]":
    return isinstance(obj, str | Callable | dict | kf.KCell)
