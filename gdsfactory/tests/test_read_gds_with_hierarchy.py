# from pprint import pprint

import jsondiff
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf

gdspath = gf.CONFIG["gdsdir"] / "mzi2x2.gds"


def test_read_gds_hash2() -> gf.Component:
    c = gf.import_gds(gdspath)
    assert (
        c.hash_geometry() == "7bc2d1d759011e5f41daf8683cf9467bdc8ebad4"
    ), c.hash_geometry()
    return c


def test_read_gds_with_settings2(data_regression: DataRegressionFixture) -> None:
    c = gf.import_gds(gdspath)
    data_regression.check(c.to_dict())


def test_read_gds_equivalent2():
    """Ensures we can load it from GDS + YAML and get the same component settings"""
    c1 = gf.c.mzi()
    c2 = gf.import_gds(gdspath)

    d1 = c1.to_dict()
    d2 = c2.to_dict()

    # we change the name, so there is no cache conflicts
    d1["info"].pop("name")
    d2["info"].pop("name")

    d1.pop("cells")
    d2.pop("cells")
    d1.pop("ports")
    d2.pop("ports")
    # c1.pprint()
    # c2.pprint()

    d = jsondiff.diff(d1, d2)

    from pprint import pprint

    # pprint(d1)
    pprint(d2)
    # pprint(d)
    assert len(d) == 0, d


def test_mix_cells_from_gds_and_from_function2():
    """Ensures not duplicated cell names.
    when cells loaded from GDS and have the same name as a function
    with @cell decorator
    """
    c = gf.Component("test_mix_cells_from_gds_and_from_function")
    c << gf.c.mzi()
    c << gf.import_gds(gdspath)
    c.write_gds()
    c.show()


def _write():
    c1 = gf.c.mzi()
    c1.name = "mzi_gds"
    c1.write_gds_with_metadata(gdspath=gdspath)
    c1.show()


if __name__ == "__main__":
    # _write()
    test_read_gds_equivalent2()

    # c = test_read_gds_hash2()
    # c.show()
    # test_mix_cells_from_gds_and_from_function2()

    # _write()
    # test_read_gds_with_settings2()
    # test_read_gds_equivalent2()

    # c1 = gf.c.mzi()
    # c2 = gf.import_gds(gdspath)
    # d1 = c1.to_dict_config()
    # d2 = c2.to_dict_config()
    # dd1 = c1.to_dict()
    # dd2 = c2.to_dict()
    # dd1["info"].pop("name")
    # dd2["info"].pop("name")
    # dd1.pop("cells")
    # dd2.pop("cells")
    # dd1.pop("ports")
    # dd2.pop("ports")
    # d = jsondiff.diff(dd1, dd2)
    # print(d)
