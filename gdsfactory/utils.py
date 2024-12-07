from __future__ import annotations

from typing import TYPE_CHECKING

import klayout.db as kdb

if TYPE_CHECKING:
    from gdsfactory.typings import BoundingBoxes


def to_boxes(bounding_boxes: "BoundingBoxes") -> list[kdb.Box]:
    return [kdb.Box(*b) for b in bounding_boxes]
