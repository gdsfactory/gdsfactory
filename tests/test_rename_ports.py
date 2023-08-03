from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


@pytest.mark.parametrize("port_type", ["electrical", "optical", "placement"])
def test_rename_ports(port_type, data_regression: DataRegressionFixture):
    c = gf.components.nxn(port_type=port_type)
    data_regression.check(c.to_dict())


if __name__ == "__main__":
    c = gf.c.nxn(port_type="placement")
    # c = gf.c.nxn(port_type="optical")
    c.show(show_ports=True)
