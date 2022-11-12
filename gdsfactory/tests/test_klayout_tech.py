from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.config import PATH
from gdsfactory.klayout_tech import LayerDisplayProperties


def test_klayout_tech_create(
    data_regression: DataRegressionFixture, check: bool = True
) -> LayerDisplayProperties:

    lyp = LayerDisplayProperties.from_lyp(str(PATH.klayout_lyp))

    if check:
        data_regression.check(lyp.dict())

    return lyp


if __name__ == "__main__":
    lyp = test_klayout_tech_create(None, check=False)
    d = lyp.dict()
