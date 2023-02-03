"""ensures gdslibrary exists."""

from __future__ import annotations

import gdsfactory as gf


def test_gdslib_exists() -> None:
    assert gf.PATH.gdslib.exists()


if __name__ == "__main__":
    test_gdslib_exists()
    # print(gf.CONFIG['gdslib'])
