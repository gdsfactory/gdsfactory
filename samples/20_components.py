"""
# Components

We store our component functions inside the `pp.components` module. Each function there returns a Component object

You can use `dir` the `pp.components` module to see the all available components.

Some of which are just shapes, but we call them components as they all inherit from the component class in `pp.Component`
"""


import pp

if __name__ == "__main__":
    print(dir(pp.c))
