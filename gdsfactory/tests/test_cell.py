import pytest
from pydantic import ValidationError

import gdsfactory as gf


@gf.cell
def _dummy(length: int = 3, wg_width: float = 0.5) -> gf.Component:
    c = gf.Component()
    return c


def test_raise_error_args():
    with pytest.raises(ValueError):
        _dummy(3)


@gf.cell
def _dummy2(length: int = 3, wg_width: float = 0.5) -> gf.Component:
    c = gf.Component()
    return c


def test_validator_error():
    with pytest.raises(ValidationError):
        _dummy2(length="error")


if __name__ == "__main__":
    # test_raise_error_args()
    test_validator_error()
