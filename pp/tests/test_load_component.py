from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

import pp
from pp.load_component import load_component


def test_load_component_gds() -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    assert c.hash_geometry() == "acbc8481cf28bc9930ecbd373cafcd17b39c5c27"


def test_load_component_ports(num_regression: NumericRegressionFixture) -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    num_regression.check(c.get_ports_array())


def test_load_component_settings(data_regression: DataRegressionFixture) -> None:
    gdspath = pp.CONFIG["gdsdir"] / "waveguide.gds"
    c = load_component(gdspath)
    data_regression.check(c.get_settings())


if __name__ == "__main__":
    test_load_component_gds()
