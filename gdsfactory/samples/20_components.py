"""# Components.

We store our component functions inside the `gdsfactory.components` module. Each function there returns a Component object

You can use `dir` the `gdsfactory.components` module to see the all available components.

Some of which are just shapes, but we call them components as they all inherit from the component class in `gdsfactory.Component`
"""


import gdsfactory


def test_components_module() -> None:
    assert len(dir(gdsfactory.components)) > 1


if __name__ == "__main__":
    print(dir(gdsfactory.components))
    print(len(dir(gdsfactory.components)))
