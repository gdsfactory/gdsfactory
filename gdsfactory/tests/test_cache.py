import pytest

import gdsfactory as gf
from gdsfactory.component import MutabilityError


def test_cache_fail(padding: float = 3.0) -> gf.Component:
    """
    Adding padding to a cached component should raise MutabilityError

    Args:
        default: default padding on all sides
    """
    c = gf.c.straight()

    with pytest.raises(MutabilityError):
        c.add_padding(default=padding)  # add padding to original cell

    return c


if __name__ == "__main__":
    c = test_cache_fail()
    c.show()
