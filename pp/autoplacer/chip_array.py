import itertools
import os

import klayout.db as pya

import pp.autoplacer.functions as ap
from pp.autoplacer.auto_placer import AutoPlacer
from pp.autoplacer.library import Library


class ChipArray(AutoPlacer):
    """ An array of chiplets with dicing lanes
    """

    def __init__(
        self,
        name: str,
        mask_width: float,
        mask_height: float,
        cols: int,
        rows: int,
        lib: Library,
        spacing: int = 25000,
        lane_width: int = 50000,
        align: None = None,
    ) -> None:
        super(ChipArray, self).__init__(name, mask_width, mask_height)
        self.lib = lib
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.lane_width = lane_width
        self.align = align

        # Infer chip width and height
        self.chip_width = (mask_width - (cols - 1) * spacing) / cols
        self.chip_height = (mask_height - (rows - 1) * spacing) / rows

        self.make_chips()
        # self.make_dicing_lanes()

    def make_chips(self) -> None:
        """ Make all the chips """
        # Get the aligntree
        if self.align:
            aligntrees = self.lib.get(self.align)
            self.aligntree = aligntrees[0]

        # Make all the chips
        self.chips = []
        for row, col in itertools.product(
            list(range(self.rows)), list(range(self.cols))
        ):
            name = "{}{}".format(row, col)
            chip = AutoPlacer(name, self.chip_width, self.chip_height)
            if self.align:
                for corner in ap.CORNERS:
                    chip.pack_auto(self.aligntree, corner)

            chip.draw_boundary()
            chip.row, chip.col = row, col
            self.chips.append(chip)

    def make_dicing_lanes(self):
        """ Make the dicing lanes """
        container = self.create_cell("DicingLanes")
        instance = pya.CellInstArray(container.cell_index(), pya.Trans(0, 0))
        self.cell(self.name).insert(instance)
        lw = self.lane_width / 2

        for dicing_layer in ap.DICING_LAYERS:
            layer = self.layer(dicing_layer[0], dicing_layer[1])

            for row in range(1, self.rows):
                y = row * (self.chip_height + self.spacing) - self.spacing / 2
                box = pya.Box(0, y - lw, self.max_width, y + lw)
                container.shapes(layer).insert(box)

            for col in range(1, self.cols):
                x = col * (self.chip_width + self.spacing) - self.spacing / 2
                for row in range(self.rows):
                    y1 = row * (self.chip_height + self.spacing)
                    y2 = (row + 1) * (self.chip_height + self.spacing) - self.spacing
                    box = pya.Box(x - lw, y1, x + lw, y2)
                    container.shapes(layer).insert(box)

            # on the corners, line has half the width
            lw = self.lane_width / 4

            for row in [0, self.rows]:
                y = row * (self.chip_height + self.spacing) - self.spacing / 2
                box = pya.Box(0, y - lw, self.max_width, y + lw)
                container.shapes(layer).insert(box)

            for col in [0, self.cols]:
                x = col * (self.chip_width + self.spacing) - self.spacing / 2
                for row in range(self.rows):
                    y1 = row * (self.chip_height + self.spacing)
                    y2 = (row + 1) * (self.chip_height + self.spacing) - self.spacing
                    box = pya.Box(x - lw, y1, x + lw, y2)
                    container.shapes(layer).insert(box)

    def write(self, *args, **kwargs) -> None:
        """ Write to disk. We pack the chips at the last minute. """
        self.draw_boundary(ap.DEVREC_LAYER)
        self.draw_boundary(ap.FLOORPLAN_LAYER)
        for chip in self.chips:
            x = chip.col * (self.chip_width + self.spacing)
            y = chip.row * (self.chip_height + self.spacing)
            self.pack_manual(chip.top_cell(), x, y)
        super(ChipArray, self).write(*args, **kwargs)

    def write_chips(self, name=None, path=None):
        if name is None:
            name = self.name
        if path is None:
            path = os.path.join("build", "mask")
        filename = os.path.join(path, name)
        for chip in self.chips:
            chip.write(filename + "_" + chip.name + ".gds")


if __name__ == "__main__":
    lib = Library()
    mask = ChipArray("chip_array", 25e6, 25e6, 3, 4, lib)
    mask.pack_grid(lib.pop("align"))
    mask.pack_grid(lib.pop(".*"))
    mask.write("build/chip_array.gds")
