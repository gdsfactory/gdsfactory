""" ensures gdslibrary exists
"""

import pp


def test_gdslib_exists() -> None:
    assert pp.CONFIG["gdslib"].exists()


if __name__ == "__main__":
    test_gdslib_exists()
    # print(pp.CONFIG['gdslib'])
