""" ensures gdslibrary exists
"""

import gdsfactory


def test_gdslib_exists() -> None:
    assert gdsfactory.CONFIG["gdslib"].exists()


if __name__ == "__main__":
    test_gdslib_exists()
    # print(gdsfactory.CONFIG['gdslib'])
