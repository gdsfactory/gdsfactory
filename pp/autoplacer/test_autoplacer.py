from pp import klive
from pp.autoplacer.chip_array import ChipArray
from pp.autoplacer.library import Library


def test_autplacer() -> None:
    lib = Library()
    mask = ChipArray("chip_array", 25e6, 25e6, 3, 4, lib)
    mask.pack_grid(lib.pop("align"))
    mask.pack_grid(lib.pop(".*"))
    mask.write("chip_array.gds")
    klive.show("chip_array.gds")


if __name__ == "__main__":
    test_autplacer()
