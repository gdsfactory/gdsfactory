from pytest_regressions.data_regression import DataRegressionFixture

import pp
from pp.component import Component


def test_route_south(
    data_regression: DataRegressionFixture, check: bool = True
) -> Component:
    c = pp.Component("test_route_south")
    cr = c << pp.components.mmi2x2()
    route = pp.routing.route_south(cr)
    references = route["references"]

    lengths = {}
    for i, reference in enumerate(references):
        c.add(reference)
        route_length = reference.parent.length
        lengths[i] = route_length
    if check:
        data_regression.check(lengths)
    return c


if __name__ == "__main__":
    c = test_route_south(None, check=False)
    c.show()
