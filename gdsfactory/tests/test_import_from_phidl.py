import phidl.geometry as pg
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_import_from_phidl(data_regression: DataRegressionFixture) -> None:
    """Read ports from markers."""

    c1 = pg.snspd()
    c2 = gf.read.from_phidl(component=c1)
    data_regression.check(c2.to_dict())


if __name__ == "__main__":
    c1 = pg.snspd()
    c2 = gf.read.from_phidl(component=c1)
    c2.show(show_ports=True)
