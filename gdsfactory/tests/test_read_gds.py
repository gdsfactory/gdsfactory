# from pprint import pprint

import jsondiff

import gdsfactory as gf


def test_read_gds_hash() -> gf.Component:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.import_gds(gdspath)
    h = "c8b69d8eb2f61eeb2a6600c27d83c227d1c3ce62"
    assert c.hash_geometry() == h, f"h = {c.hash_geometry()!r}"
    return c


# def test_read_gds_with_settings(data_regression: DataRegressionFixture) -> None:
#     gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
#     c = gf.import_gds(gdspath)
#     data_regression.check(c.to_dict())


def test_read_gds_equivalent() -> None:
    """Ensures we load Component from GDS + YAML and get the same component
    settings."""
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"

    c2 = gf.import_gds(gdspath)
    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d = jsondiff.diff(d1, d2)

    # pprint(d1)
    # pprint(d2)
    # pprint(d)
    assert len(d) == 0, d


def test_mix_cells_from_gds_and_from_function() -> None:
    """Ensures not duplicated cell names.

    when cells loaded from GDS and have the same name as a function with
    @cell decorator
    """
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.Component("test_mix_cells_from_gds_and_from_function")
    c << gf.components.straight(length=1.234)
    c << gf.import_gds(gdspath)
    c.write_gds()


def _write() -> None:
    c1 = gf.components.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c1.write_gds_with_metadata(gdspath=gdspath)
    c1.show()
    c1.pprint()


if __name__ == "__main__":
    # _write()

    # test_mix_cells_from_gds_and_from_function()
    # test_read_gds_equivalent()
    test_read_gds_hash()

    # c1 = gf.components.straight(length=1.234)
    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"

    # c2 = gf.import_gds(gdspath, name="c2")
    # d = c2.to_dict()["cells"]
    # print(d)

    # c1 = gf.components.straight(length=1.234)
    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    # c2 = gf.import_gds(gdspath)
    # d1 = c1.to_dict()
    # d2 = c2.to_dict()

    # d = jsondiff.diff(d1, d2)
    # assert len(d) == 0, d
