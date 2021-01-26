from pytest_regressions.data_regression import DataRegressionFixture

from pp.difftest import difftest
from pp.routing.route_ports_to_side import sample_route_sides


def test_sample_route_sides(data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    c = sample_route_sides()
    difftest(c)
