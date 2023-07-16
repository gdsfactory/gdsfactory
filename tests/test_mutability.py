from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.component import MutabilityError


def test_mutability() -> None:
    """Adding padding to a cached component should raise MutabilityError."""
    c = gf.components.straight()

    with pytest.raises(MutabilityError):
        c.add_padding(default=3)  # add padding to original cell


if __name__ == "__main__":
    test_mutability()
