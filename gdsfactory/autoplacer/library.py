import glob
import json
import re
from collections import defaultdict

import klayout.db as pya

from gdsfactory.autoplacer.cell_list import CellList
from gdsfactory.autoplacer.functions import WORKING_MEMORY, area


class Library(object):
    """Library of cells with convenient methods to:

        - load from disk
        - pop groups
        - add padding

    Args:
        root: GDS devices path

    To make a `Library` containing all the devices in `build/devices`, just instantiate the class (`library = Library()`).
    You can then pull out subsets of devices using `library.pop()`.
    `library.pop()` accepts a regular expression, which is matched against the top-cell name, to select the desired cells.
    It also has `padding` and `normalize` arguments.

    By default, padding is added to cells when they are popped from the `Library`.
    Attempts are also made to normalize the form-factor of the devices so that the grating couplers are aligned.
    This will come at the cost of some space.
    If you are packing many disparate components, you should either group them into blocks first, or pass `normalize=False`.

    .. code::

        import gdsfactory.autoplacer as ap

        # Load all the GDS files from build/devices
        library = ap.Library()

        # Select all devices with "ring" and "euler" in the top-cell name.
        library.pop("ring.*euler.*")

        # Same selection, less padding
        library.pop("ring.*euler.*", padding=0)

        # Same selection, without attempting to align gratings
        library.pop("ring.*euler.*", normalize=False)

    """

    def __init__(self, root: str = "build/devices") -> None:
        self.root = root
        self.cells = {}
        self.does = defaultdict(list)
        self.load_all_gds()
        self.load_all_json()

    def load_all_gds(self) -> None:
        """Loads all the gds files"""
        filenames = glob.glob(self.root + "/*.gds")
        print("Loading {} GDS files...".format(len(filenames)))
        for filename in filenames:
            self.load_gds(filename)
        print("Done")

    def load_all_json(self) -> None:
        """loads all the json files"""
        filenames = glob.glob(self.root + "/*.json")
        for filename in filenames:
            self.load_json(filename)

    def load_gds(self, filename):
        """Load a GDS and append it into self.cells"""
        layout = pya.Layout()
        WORKING_MEMORY[filename] = layout
        layout.read(str(filename))
        self.cells[layout.top_cell().name] = layout.top_cell()
        self.cells[layout.top_cell().name].metadata = {}

    def load_json(self, filename):
        """Load json metadata"""
        with open(filename) as f:
            metadata = json.load(f)
            if metadata.get("type") == "sweep":
                doe_name = metadata.get("name")
                for cell_name in metadata.get("cells"):
                    if cell_name in self.cells:
                        self.does[doe_name].append(self.cells[cell_name])

    def get(self, regex):
        cells = [
            cell
            for key, cell in self.cells.items()
            if re.search(regex, key, flags=re.IGNORECASE)
        ]
        return CellList(cells)

    def pop_doe(self, regex):
        """pop out a set of cells"""
        cells = []
        if regex in self.does:
            cells = self.does[regex]
            del self.does[regex]
            self.delete_cells(cells)
            cells = sorted(cells, key=area, reverse=True)

        else:
            print("Warning: no cells found for {}".format(regex))

        return CellList(cells)

    def pop(self, regex: str, delete: bool = True) -> CellList:
        """pop cells"""
        # pop out the cells
        # cells = [
        #     cell for key, cell in self.cells.items()
        #     if re.search(regex, key, flags=re.IGNORECASE)
        # ]
        cells = []
        keys = []
        for key, cell in self.cells.items():
            if re.search(regex, key, flags=re.IGNORECASE):
                cells.append(cell)
                keys.append(key)

        if delete:
            for key in keys:
                del self.cells[key]
        if cells:
            cells = sorted(cells, key=area, reverse=True)
        else:
            print("Warning: no cells found for {}".format(regex))

        return CellList(cells)

    def delete_cells(self, cells):
        for cell in cells:
            try:
                del self.cells[cell.name]
            except KeyError:
                pass

    def list(self):
        """just list the devices currently in the collection"""
        print("Library contains cells:")
        for name in sorted(self.cells.keys()):
            print("-", name)

        if self.does and False:
            print()
            print("DOEs:")
            for key, value in self.does.items():
                print(" - {} ({})".format(key, len(value)))

    def count(self):
        """Safety check at the end"""
        if self.cells:
            print("{} cells were not used".format(len(self.cells)))

    def __str__(self):
        return "<collection of {} cells>".format(len(self.cells))


if __name__ == "__main__":
    lib = Library()
