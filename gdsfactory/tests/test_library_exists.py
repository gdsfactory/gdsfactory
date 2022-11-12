"""ensures gdslibrary exists."""

import gdsfactory as gf


def test_gdslib_exists() -> None:
    assert gf.CONFIG["gdslib"].exists()


if __name__ == "__main__":
    test_gdslib_exists()
    # print(gf.CONFIG['gdslib'])
