try:
    from gdsfactory.autoplacer.chip_array import ChipArray
    from gdsfactory.autoplacer.library import Library

except ModuleNotFoundError:
    print(
        "klayout installation not found. You need to `pip install klayout` to use the klayout placer"
    )


def test_autplacer():

    lib = Library()
    mask = ChipArray("chip_array", 25e6, 25e6, 3, 4, lib)
    mask.pack_grid(lib.pop("align"))
    mask.pack_grid(lib.pop(".*"))
    mask.write("chip_array.gds")


if __name__ == "__main__":
    import gdsfactory as gf

    test_autplacer()
    gf.show("chip_array.gds")
