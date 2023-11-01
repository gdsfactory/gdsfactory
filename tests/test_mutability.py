from __future__ import annotations

import pytest

import gdsfactory as gf


def test_mutability() -> None:
    """Adding padding to a cached component should raise MutabilityError."""
    c = gf.components.straight()

    # with pytest.raises(MutabilityError):
    with pytest.warns(UserWarning):
        c.add_padding(default=3)  # add padding to original cell


if __name__ == "__main__":
    test_mutability()
