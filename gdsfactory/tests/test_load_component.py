from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

import gdsfactory as gf


def test_load_component_gds() -> None:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.gds(gdspath)
    assert c.hash_geometry() == "4b8f6646dcf60b78b905ac0c1665a35f119be32a"


def test_load_component_ports(num_regression: NumericRegressionFixture) -> None:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.gds(gdspath)
    num_regression.check(c.get_ports_array())


def test_load_component_settings(data_regression: DataRegressionFixture) -> None:
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.gds(gdspath)
    data_regression.check(c.get_settings())


if __name__ == "__main__":
    test_load_component_gds()
