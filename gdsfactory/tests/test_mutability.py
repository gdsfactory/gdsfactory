import pytest

import gdsfactory as gf
from gdsfactory.component import MutabilityError


def test_mutability() -> gf.Component:
    """Adding padding to a cached component should raise MutabilityError."""
    c = gf.components.straight()

    with pytest.raises(MutabilityError):
        c.add_padding(default=3)  # add padding to original cell

    return c


if __name__ == "__main__":
    c = test_mutability()
    c.show(show_ports=True)
