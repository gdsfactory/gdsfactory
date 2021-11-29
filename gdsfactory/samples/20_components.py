"""# Components.
You can adapt some component functions from the `gdsfactory.components` module.
Each function there returns a Component object
"""


import gdsfactory as gf


def test_components_module() -> None:
    assert len(dir(gf.components)) > 1


if __name__ == "__main__":
    print(dir(gf.components))
    print(len(dir(gf.components)))
