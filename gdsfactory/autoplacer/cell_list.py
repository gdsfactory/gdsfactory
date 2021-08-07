import math
from collections import defaultdict
from typing import Any, List

import klayout.db as pya

import gdsfactory.autoplacer.functions as ap


class CellList:
    """just a list of components,
    with the ability to normalize size and shape"""

    def __init__(self, cells: List[Any]) -> None:
        self.cells = cells

    def find_origin(self, cell):
        """finds the displacement of the origin"""
        origins = []
        for instance in cell.each_inst():
            if instance.cell.name.endswith("t_c"):
                origins.append(instance.trans.disp)

        if origins:
            return pya.Point(
                ap.mean([o.x for o in origins]), ap.mean([o.y for o in origins])
            )
        else:
            bbox = cell.bbox()
            return pya.Point(-bbox.left, -bbox.bottom)

    def align(self, dimension=ap.BOTH):
        """Aligns the grating couplers"""
        if not self.cells:
            return

        for cell in self.cells:
            cell.origin = self.find_origin(cell)

        max_origin = pya.Point(
            max(cell.origin.x for cell in self.cells),
            max(cell.origin.y for cell in self.cells),
        )

        for cell in self.cells:
            delta = max_origin - cell.origin
            layer = cell.layout().layer(ap.DEVREC_LAYER, 0)
            delta.x = delta.x if dimension in (ap.BOTH, ap.WIDTH) else 0
            delta.y = delta.y if dimension in (ap.BOTH, ap.HEIGHT) else 0
            box = cell.bbox().enlarge(delta.x, delta.y)
            cell.shapes(layer).insert(box)

    def normalize(self, dimension, origin=ap.SOUTH_WEST):
        """Makes all components the same size on some dimension"""
        if not self.cells:
            return

        max_width = max(cell.bbox().width() for cell in self.cells)
        max_height = max(cell.bbox().height() for cell in self.cells)
        for cell in self.cells:
            bbox = cell.bbox()
            if not (bbox.width() == max_width and bbox.height == max_height):
                dx = max_width - bbox.width() if dimension in (ap.WIDTH, ap.BOTH) else 0
                dy = (
                    max_height - bbox.height()
                    if dimension in (ap.HEIGHT, ap.BOTH)
                    else 0
                )

                if origin == ap.SOUTH_WEST:
                    box = pya.Box(
                        bbox.left, bbox.bottom, bbox.right + dx, bbox.top + dy
                    )
                elif origin == ap.SOUTH_EAST:
                    box = pya.Box(
                        bbox.left - dx, bbox.bottom, bbox.right, bbox.top + dy
                    )
                elif origin == ap.NORTH_WEST:
                    box = pya.Box(
                        bbox.left, bbox.bottom - dy, bbox.right + dx, bbox.top
                    )
                elif origin == ap.NORTH_EAST:
                    box = pya.Box(
                        bbox.left - dx, bbox.bottom - dy, bbox.right, bbox.top
                    )

                layer = cell.layout().layer(ap.DEVREC_LAYER, 0)
                cell.shapes(layer).insert(box)

    def pad(self, padding=ap.PADDING):
        """add padding to a cell"""
        for cell in self.cells:
            layer = cell.layout().layer(ap.DEVREC_LAYER, 0)
            bbox = cell.bbox()
            cell.shapes(layer).insert(bbox)
            box = pya.Box(
                bbox.left - padding,
                bbox.bottom - padding,
                bbox.right + padding,
                bbox.top + padding,
            )
            cell.shapes(layer).insert(box)

    def sort(self, key):
        """Sort the cells"""
        self.cells = sorted(self.cells, key=key)

    def groups(self, dimension, granularity=ap.GRANULARITY):
        """group cells by size"""
        groups = defaultdict(list)
        for cell in self.cells:
            b = cell.bbox()
            lookup = {ap.WIDTH: b.width(), ap.HEIGHT: b.height(), ap.AREA: b.area()}
            size = lookup[dimension]
            size = granularity * math.floor(size // granularity)
            groups[size].append(cell)
        return [
            CellList(sorted(groups[key], key=ap.area))
            for key in sorted(list(groups.keys()), reverse=True)
        ]

    def __iter__(self):
        """allow for cell in celllist"""
        for cell in sorted(self.cells, key=ap.area):
            yield cell

    def __getitem__(self, index):
        """allow for cell in celllist"""
        return self.cells[index]

    def __len__(self) -> int:
        return len(self.cells)

    def __str__(self):
        s = "\n".join(" - " + c.name for c in self.cells[:3])
        if len(self.cells) > 3:
            s += "\n..."
        return "list of {} cells:\n{}".format(len(self.cells), s)
