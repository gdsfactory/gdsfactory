from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import klayout.db as kdb

if TYPE_CHECKING:
    from gdsfactory.typings import BoundingBox


def to_kdb_boxes(bounding_boxes: "Sequence[BoundingBox | kdb.Box]") -> list[kdb.Box]:
    return [
        box if isinstance(box, kdb.Box) else kdb.Box(*map(int, box))
        for box in bounding_boxes
    ]
