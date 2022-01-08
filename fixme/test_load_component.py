import jsondiff
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_load_component_gds() -> gf.Component:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.import_gds(gdspath)
    assert c.hash_geometry() == "4b8f6646dcf60b78b905ac0c1665a35f119be32a"
    return c


def test_load_component_settings(data_regression: DataRegressionFixture) -> None:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.import_gds(gdspath)
    data_regression.check(c.to_dict())


def test_load_component_with_settings():
    """Ensures we can load it from GDS + YAML and get the same component settings"""
    c1 = gf.c.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"

    c2 = gf.import_gds(gdspath)

    d1 = c1.to_dict()
    d2 = c2.to_dict()
    d = jsondiff.diff(d1, d2)
    assert len(d) == 0, d


def _write_gds():
    c = gf.c.straight(length=1.234)
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c.write_gds_with_metadata(gdspath)


if __name__ == "__main__":
    # _write_gds()
    # test_load_component_gds()
    # test_load_component_settings()
    test_load_component_with_settings()

    # c1 = gf.c.straight()
    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"

    # c2 = gf.import_gds(gdspath)
    # d = jsondiff.diff(c1.to_dict, c2.to_dict())
    # print(d)
