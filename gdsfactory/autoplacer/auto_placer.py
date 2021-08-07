import itertools
import math
from typing import Tuple

import klayout.db as pya
import pyqtree
from klayout.dbcore import Cell

import gdsfactory.autoplacer.functions as ap
from gdsfactory.autoplacer.cell_list import CellList
from gdsfactory.autoplacer.library import Library


class AutoPlacer(pya.Layout):
    """klayout autoplacer

    Args:
        name: name of the container
        max_width: 1e8 (nm)
        max_height: 1e8 (nm)

    `AutoPlacer` is the class which knows how to automatically pack and position cells. You feed it cells that are loaded and padded using `library`.

    Instantiate an AutoPlacer with a `name`, `max_width` and `max_height` properties. If this Autoplacer is the main top-level cell of the mask, `max_width` and `max_height` should be the desired width and height of the mask.

    You can then pack cells into the AutoPlacer using the `AutoPlacer.pack_auto`, `AutoPlacer.pack_manual`, and `AutoPlacer.pack_many` methods.

    - `AutoPlacer.pack_auto(cell, origin, direction)` automatically finds a location for the component. You can provide an `origin` argument, which can take any value from `autoplacer.SOUTH_WEST`, `autoplacer.SOUTH_EAST`, `autoplacer.NORTH_WEST`, `autoplacer.NORTH_EAST`. The autoplacer will search for a place to put the cell starting from that cardinal point. It can search in either `VERTICAL` or `HORIZONTAL` direction.
    - `AutoPlacer.pack_manual(cell, x, y, origin)` allows you to manually place a component at an absolute location.
    - `AutoPlacer.pack_many` behaves like `AutoPlacer.pack_auto` but accepts a list of devices.

    """

    def __init__(
        self, name: str, max_width: float = 1e8, max_height: float = 1e8
    ) -> None:
        """constructor"""
        # Construct
        super(AutoPlacer, self).__init__()

        # get width and height
        self.max_width = max_width
        self.max_height = max_height
        global COUNTER, AUTOPLACER_REGISTRY
        if name in ap.AUTOPLACER_REGISTRY:
            self.name = "{}_{}".format(name, ap.COUNTER)
            ap.COUNTER += 1
        else:
            self.name = name
        ap.AUTOPLACER_REGISTRY[self.name] = self

        # Register
        ap.WORKING_MEMORY["__AutoPlacer_{}".format(self.name)] = self

        # Create the quadtree which will enable efficient queries
        bbox = (0, 0, self.max_width, self.max_height)
        self.quadtree = pyqtree.Index(bbox=bbox)

        # Make a topcell
        self.create_cell(self.name)

    def get_edge(self, collisions, bbox, direction):
        """Get a particular edge of some collisions"""
        if direction == ap.NORTH:
            return max(n for (w, s, e, n) in collisions) + ap.GRID
        elif direction == ap.SOUTH:
            return min(s for (w, s, e, n) in collisions) - ap.GRID - bbox.height()
        elif direction == ap.EAST:
            return min(e for (w, s, e, n) in collisions) + ap.GRID
        elif direction == ap.WEST:
            return max(w for (w, s, e, n) in collisions) - ap.GRID - bbox.width()

    def import_cell(self, cell: Cell) -> Cell:
        """Imports a cell from another Layout"""
        # If the cell is already in the library, skip loading
        if self.cell(cell.name):
            return self.cell(cell.name)

        # Create a holder cell and copy in the shapes
        new_cell = self.create_cell(cell.name)
        new_cell.copy_shapes(cell)

        # Import all the child cells
        for child_index in cell.each_child_cell():
            self.import_cell(cell.layout().cell(child_index))

        # Import all of the instances, doing the mapping from Layout to Layout
        for instance in cell.each_inst():
            cell_index = self.cell(instance.cell.name).cell_index()
            # new_instance = pya.CellInstArray(cell_index, instance.trans)
            new_instance = instance.cell_inst.dup()
            new_instance.cell_index = cell_index
            new_cell.insert(new_instance)
        return new_cell

    def inside(self, bbox):
        """Check that something is inside the mask"""
        mask = pya.Box(0, 0, self.max_width + 1, self.max_height + 1)
        return mask.contains(bbox)

    def find_collisions(self, bbox, sx, sy):
        """Test for intersection of objects"""
        box = sx, sy, sx + bbox.width(), sy + bbox.height()
        return self.quadtree.intersect(box)

    def compute_start(self, cell, origin):
        """Figure out where the search should start for a given cell"""
        w = cell.bbox().width()
        h = cell.bbox().height()
        if origin == ap.SOUTH_WEST:
            return [0, 0]
        elif origin == ap.SOUTH_EAST:
            return [self.max_width - w, 0]
        elif origin == ap.NORTH_WEST:
            return [0, self.max_height - h]
        elif origin == ap.NORTH_EAST:
            return [self.max_width - w, self.max_height - h]

    def find_space(self, cell, origin=ap.SOUTH_WEST, direction=ap.VERTICAL):
        """Find space for a cell by brute-force search"""
        if direction == ap.VERTICAL:
            return self.find_space_vertical(cell, origin)
        elif direction == ap.HORIZONTAL:
            return self.find_space_horizontal(cell, origin)

    def find_space_vertical(self, cell, origin=ap.SOUTH_WEST):
        """Find space for a cell by brute-force search - horizontal"""
        # Compute boundaries
        bbox = cell.bbox()
        x_bound = self.max_width - bbox.width()
        y_bound = self.max_height - bbox.height()

        # Compute start position
        start = self.compute_start(cell, origin)
        sx = start[0]
        while sx >= 0 and sx <= x_bound:
            sy = start[1]
            edge = []
            while sy >= 0 and sy <= y_bound:
                # Find any collisions with cell that have been packed
                collisions = self.find_collisions(bbox, sx, sy)
                if len(collisions) == 0:
                    return sx, sy

                # Otherwise, search from the northmost collision
                sy = self.get_edge(
                    collisions, bbox, ap.NORTH if ap.SOUTH in origin else ap.SOUTH
                )
                e = self.get_edge(
                    collisions, bbox, ap.EAST if ap.WEST in origin else ap.WEST
                )
                edge.append(e)

            if not edge:
                return
            sx = min(edge) if ap.WEST in origin else max(edge)

    def find_space_horizontal(self, cell, origin=ap.SOUTH_WEST):
        """Find space for a cell by brute-force search - vertical"""
        # Compute boundaries
        bbox = cell.bbox()
        x_bound = self.max_width - bbox.width()
        y_bound = self.max_height - bbox.height()

        # Compute start position
        start = self.compute_start(cell, origin)
        sy = start[1]
        while sy >= 0 and sy <= y_bound:
            sx = start[0]
            edge = []
            while sx >= 0 and sx <= x_bound:
                # Find any collisions with cell that have been packed
                collisions = self.find_collisions(bbox, sx, sy)
                if len(collisions) == 0:
                    return sx, sy

                # Otherwise, search from the northmost collision
                sx = self.get_edge(
                    collisions, bbox, ap.EAST if ap.WEST in origin else ap.WEST
                )
                e = self.get_edge(
                    collisions, bbox, ap.NORTH if ap.SOUTH in origin else ap.SOUTH
                )
                edge.append(e)

            if not edge:
                return
            sy = min(edge) if ap.SOUTH in origin else max(edge)

    def pack_manual(
        self,
        cell: Cell,
        x: float,
        y: float,
        origin: Tuple[int, int] = ap.SOUTH_WEST,
        tboxes: None = None,
    ) -> None:
        """
        Pack a cell at a manually selected position
        """
        box = cell.bbox()

        # Do an offset if origin is given
        w = box.width()
        h = box.height()
        oy, ox = origin
        x -= {ap.WEST: 0, ap.EAST: w, ap.MIDDLE: w / 2}[ox]
        y -= {ap.SOUTH: 0, ap.NORTH: w, ap.MIDDLE: h / 2}[oy]

        # Insert into the quadtree

        if tboxes is None:
            tbox = (x, y, x + box.width(), y + box.height())
            tboxes = [tbox]

        for tbox in tboxes:
            self.quadtree.insert(tbox, tbox)

        new_cell = self.import_cell(cell)

        # Make an instance
        transform = pya.Trans(int(x - box.left), int(y - box.bottom))
        new_instance = pya.CellInstArray(new_cell.cell_index(), transform)
        self.cell(self.name).insert(new_instance)

    def pack_auto(self, cell, origin=ap.SOUTH_WEST, direction=ap.VERTICAL):
        """
        Pack a cell automatically
        """
        coordinate = self.find_space(cell, origin, direction)
        if coordinate is None:
            raise ap.OutOfSpaceError(
                "Out of space for {} ({} x {} mm) in {}".format(
                    cell.name,
                    cell.bbox().width() / 1e6,
                    cell.bbox().height() / 1e6,
                    self.name,
                )
            )
        else:
            self.pack_manual(cell, *coordinate)

    def pack_many(self, cells, origin=ap.SOUTH_WEST, direction=ap.VERTICAL):
        """
        Pack many cells with the same settings
        """
        failed = []
        for cell in cells:
            try:
                self.pack_auto(cell, origin, direction)
            except ap.OutOfSpaceError as e:
                print(e)
                failed.append(cell)
        return CellList(failed)

    def shrink(self):
        """Shrink-wrap"""
        bbox = self.top_cell().bbox()
        self.max_width = bbox.width()
        self.max_height = bbox.height()

    def draw_boundary(self, layer: int = ap.DEVREC_LAYER) -> None:
        """Draw a box into the topcell"""
        layer = self.layer(layer, 0)
        self.cell(self.name).shapes(layer).insert(
            pya.Box(0, 0, self.max_width, self.max_height)
        )

    def write(self, *args, **kwargs) -> None:
        """Draw boundary on write"""
        if not kwargs.get("shrink", False):
            self.draw_boundary()

        super(AutoPlacer, self).write(*args, **kwargs)

    """
    Below here we have high-level utility functions
    """

    def pack_groups(
        self,
        cells,
        cols=None,
        rows=None,
        aspect=3,
        dimension=ap.WIDTH,
        normalization_origin=ap.SOUTH_WEST,
        direction=ap.VERTICAL,
        granularity=ap.GRANULARITY,
        name=None,
        origin=ap.SOUTH_WEST,
    ):
        """Pack column-wise"""
        for group in cells.groups(dimension, granularity):
            self.pack_grid(
                group,
                cols,
                rows,
                aspect=aspect,
                normalization_origin=normalization_origin,
                name=name,
                origin=origin,
            )

    def pack_grid(
        self,
        cells: CellList,
        cols: None = None,
        rows: None = None,
        aspect: int = 3,
        direction: int = ap.VERTICAL,
        align: int = ap.BOTH,
        normalization_origin: Tuple[int, int] = ap.SOUTH_WEST,
        origin: Tuple[int, int] = ap.SOUTH_WEST,
        name: None = None,
        padding: int = ap.PADDING,
    ) -> None:
        """
        Pack onto a grid, assuming that all cells are the same size
        """
        if not cells:
            return

        # Get the name from the longest common substring
        if name is None:
            name = ap.longest_common_prefix([c.name for c in cells]) + "_g"
            name = name if len(name) > 2 else "Misc"

        # Make the block with approriate size
        block = AutoPlacer(name)

        # Collect and clean the cells
        cells.align(align)
        cells.normalize(ap.BOTH, normalization_origin)
        cells.pad(padding=padding)
        # cells.sort(lambda c: c.name)

        # Safety check
        assert not (
            cols is not None and rows is not None
        ), "Don't specify both rows and cols"

        # Infer the dimensions, targeting an aspect ratio of 1
        if rows is None and cols is None:
            b = cells[0].bbox()
            cols = ap.estimate_cols(len(cells), b.width(), b.height(), aspect)

        # Accept user parameters
        if cols is None:
            cols = int(math.ceil(len(cells) / rows))
        elif rows is None:
            rows = int(math.ceil(len(cells) / cols))

        # Get dimensions
        bbox = cells[0].bbox()
        w, h = bbox.width(), bbox.height()

        # Place
        assert cols * rows >= len(cells)
        grid = itertools.product(list(range(int(cols))), list(range(int(rows))))
        for cell, (col, row) in zip(cells, grid):
            block.pack_manual(cell, w * col, h * row)

        # Shrink and add a boundary
        block.shrink()
        block.draw_boundary()

        # Pack into the container
        self.pack_auto(block.top_cell(), direction=direction, origin=origin)

    def pack_random(
        self,
        cells,
        origin=ap.SOUTH_WEST,
        direction=ap.HORIZONTAL,
        normalization_origin=ap.SOUTH_WEST,
        add_padding=True,
        padding=ap.PADDING,
    ):
        """Pack at random"""
        if add_padding:
            cells.pad(padding=padding)
        return self.pack_many(cells, direction=direction, origin=origin)

    def pack_lumped(self, cells, width, height, origin=ap.SOUTH_WEST):
        """Pack at random in a lump"""
        # Get the name from the longest common substring
        name = ap.longest_common_prefix([c.name for c in cells]) + "_l"
        name = name if len(name) > 2 else "Misc"

        # Make an autoplacer
        lump = AutoPlacer(name, width, height)
        lump.pack_random(cells)
        lump.shrink()
        lump.draw_boundary()
        return self.pack_auto(lump.top_cell(), origin=origin)

    def pack_corners(self, component):
        for corner in ap.CORNERS:
            self.pack_auto(component, corner)

    def estimate_size(self, cells):
        """Estimate the size of the placer"""
        # Compute squarish dimensions
        N = len(cells)
        rows = math.ceil(math.sqrt(N))
        cols = math.floor(math.sqrt(N))

        # Analyze BBoxes
        bboxes = [cell.bbox() for cell in cells]
        max_width = max(b.width() for b in bboxes)
        max_height = max(b.height() for b in bboxes)

        # Set new max height and width
        return max_width * cols + ap.PADDING, max_height * rows + ap.PADDING


if __name__ == "__main__":
    lib = Library()
    mask = AutoPlacer("mask", 25e6, 25e6)
    mask.pack_grid(lib.pop("align"))
    mask.pack_grid(lib.pop(".*"))
    mask.write("build/mask.gds")
