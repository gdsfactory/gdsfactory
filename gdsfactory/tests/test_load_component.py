# from pprint import pprint

import jsondiff
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf

# def test_load_component_gds() -> gf.Component:
#     gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
#     c = gf.read.from_gds(gdspath)
#     assert c.hash_geometry() == "4b8f6646dcf60b78b905ac0c1665a35f119be32a"
#     return c


def test_load_component_settings(data_regression: DataRegressionFixture) -> None:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.from_gds(gdspath)
    data_regression.check(c.to_dict)


def test_load_component_with_settings():
    """Ensures we can load it from GDS + YAML and get the same component settings"""
    c1 = gf.c.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c2 = gf.read.from_gds(gdspath)

    d1 = c1.to_dict
    d2 = c2.to_dict

    # d1.pop('cells')
    # d2.pop('cells')
    # c1.pprint
    # c2.pprint

    d = jsondiff.diff(d1, d2)

    # pprint(d1)
    # pprint(d2)
    # pprint(d)
    assert len(d) == 0, d


def _write():
    c1 = gf.c.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c1.write_gds_with_metadata(gdspath=gdspath)
    c1.show()
    c1.pprint


if __name__ == "__main__":
    # _write()
    # test_load_component_gds()
    # test_load_component_settings()
    test_load_component_with_settings()

    # c1 = gf.c.straight()
    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"

    # c2 = gf.read.from_gds(gdspath)
    # d1 = c1.to_dict_config
    # d2 = c2.to_dict_config
    # dd1 = c1.to_dict
    # dd2 = c2.to_dict
    # d = jsondiff.diff(dd1, dd2)
    # print(d)
