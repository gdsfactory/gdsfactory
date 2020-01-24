"""autoplacer - placing gds components"""


from pp.autoplacer.auto_placer import AutoPlacer
from pp.autoplacer.library import Library
from pp.autoplacer.chip_array import ChipArray
from pp.autoplacer.cell_list import CellList
from pp.autoplacer.yaml_placer import place_from_yaml

__all__ = ["AutoPlacer", "Library", "CellList", "ChipArray", "place_from_yaml"]


if __name__ == "__main__":
    from pp import klive

    lib = Library()
    mask = ChipArray("chip_array", 25e6, 25e6, 3, 4, lib)
    mask.pack_grid(lib.pop("align"))
    mask.pack_grid(lib.pop(".*"))
    mask.write("chip_array.gds")
    klive.show("chip_array.gds")
