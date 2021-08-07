"""# Components.

We store our component functions inside the `gf.components` module. Each function there returns a Component object

You can use `dir` the `gf.components` module to see the all available components.

Some of which are just shapes, but we call them components as they all inherit from the component class in `gf.Component`
"""


import gdsfactory as gf


def test_components_module() -> None:
    assert len(dir(gf.components)) > 1


if __name__ == "__main__":
    print(dir(gf.components))
    print(len(dir(gf.components)))
