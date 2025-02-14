from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

import kfactory as kf
import klayout.db as kdb

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentSpec, Coordinate


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
