from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.difftest import difftest


def test_import_from_phidl(data_regression: DataRegressionFixture) -> None:
    """Read ports from markers."""
    import phidl.geometry as pg

    c1 = pg.ytron_round()
    c2 = gf.read.from_phidl(component=c1)
    data_regression.check(c2.to_dict())
    difftest(c2)


if __name__ == "__main__":
    import phidl.geometry as pg

    c1 = pg.ytron_round()
    c2 = gf.read.from_phidl(component=c1)
    c2.show(show_ports=True)
