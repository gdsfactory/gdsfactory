# # Python intro
#
# gdsfactory is written in python and requires some basic knowledge of python.
#
# If you are new to python you can find many resources online
#
# - [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
# - [youTube videos](https://www.youtube.com/c/anthonywritescode)
# - [courses](https://github.com/joamatab/practical-python)
#
# This notebook is for you to experiment with some common python patterns in `gdsfactory`
#
# ## Classes
#
# Gdsfactory has already some pre-defined classes for you.
#
# All the other classes (Component, ComponentReference, Port ...) are already available in `gf.typings`
#
# Classes are good for keeping state, which means that they store some information inside them (polygons, ports, references ...).
#
# In gdsfactory you will write functions instead of classes. Functions are easier to write and combine, and have clearly defined inputs and outputs.

# +
from pydantic import validate_arguments
from functools import partial

import gdsfactory as gf

PDK = gf.get_generic_pdk()
PDK.activate()
# -

c = gf.Component(name="my_fist_component")
c.add_polygon([(-8, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0))
c.plot()


# ## Functions
#
# Functions have clear inputs and outputs, they usually accept some parameters (strings, floats, ints ...) and return other parameters
#


# +
def double(x):
    return 2 * x


x = 1.5
y = double(x)
print(y)
# -

# It's also nice to add `type annotations` to your functions to clearly define what are the input/output types (string, int, float ...)
#


def double(x: float) -> float:
    return 2 * x


# ## Factories
#
# A factory is a function that returns an object. In gdsfactory many functions return a `Component` object
#


# +
def bend(radius: float = 5) -> gf.typings.Component:
    return gf.components.bend_euler(radius=radius)


component = bend(radius=10)

print(component)
component.plot()
# -

component


# ## Decorators
#
# gdsfactory has many functions, and we want to do some common operations for the ones that return a Component:
#
# - give a unique name (dependent on the input parameters) to a Component
# - validate input arguments based on type annotations
# - cache the Component that the function returns for speed and reuse cells.
#
# For that you will see a `@cell` decorator on many component functions.
#
# The validation functionality comes from the [pydantic](https://pydantic-docs.helpmanual.io/) package
# and is available to you automatically when using the `@cell` decorator
#


# +
@validate_arguments
def double(x: float) -> float:
    return 2 * x


x = 1.5
y = double(x)
print(y)
# -

# The validator decorator is equivalent to running
#


# +
def double(x: float) -> float:
    return 2 * x


double_with_validator = validate_arguments(double)
x = 1.5
y = double_with_validator(x)
print(y)
# -

# The `cell` decorator also leverages that validate arguments.
# So you should add type annotations to your component factories.
#
# Lets try to create an error `x` and you will get a clear message the the function `double` does not work with strings

# ```python
# y = double("not_valid_number")
# ```
#
# will raise a `ValidationError`
#
# ```
# ValidationError: 0 validation error for Double
# x
#   value is not a valid float (type=type_error.float)
#
# ```
#
# It will also `cast` the input type based on the type annotation. So if you pass an `int` it will convert it to `float`

x = 1
y = double_with_validator(x)
print(y, type(x), type(y))

# ## List comprehensions
#
# You will also see some list comprehensions, which are common in python.
#
# For example, you can write many loops in one line

# +
y = []
for x in range(3):
    y.append(double(x))

print(y)
# -

y = [double(x) for x in range(3)]  # much shorter and easier to read
print(y)


# ## Functional programming
#
# Functional programming follows linux philosophy:
#
# - Write functions that do one thing and do it well.
# - Write functions to work together.
# - Write functions with clear **inputs** and **outputs**
#
# ### partial
#
# Partial is an easy way to modify the default arguments of a function. This is useful in gdsfactory because we define PCells using functions.
#
# `gdsfactory.partial` comes from the module `functools.partial`, which is available in the standard python library.
#
# The following two functions are equivalent in functionality.
#
# Notice how the second one is shorter, more readable and easier to maintain thanks to `partial`
#


# +
def ring_sc(gap=0.3, **kwargs):
    return gf.components.ring_single(gap=gap, **kwargs)


ring_sc = partial(gf.components.ring_single, gap=0.3)
# -

# As you customize more parameters, it's more obvious that the second one is easier to maintain
#


# +
def ring_sc(gap=0.3, radius=10, **kwargs):
    return gf.components.ring_single(gap=gap, radius=radius, **kwargs)


ring_sc = partial(gf.components.ring_single, gap=0.3, radius=10)
# -

# ### compose
#
# `gf.compose` combines two functions into one.

# +
ring_sc = partial(gf.components.ring_single, radius=10)
add_gratings = gf.routing.add_fiber_array

ring_sc_gc = gf.compose(add_gratings, ring_sc)
ring_sc_gc5 = ring_sc_gc(radius=5)
ring_sc_gc5
# -

ring_sc_gc20 = ring_sc_gc(radius=20)
ring_sc_gc20

# This is equivalent and more readable than writing

ring_sc_gc5 = add_gratings(ring_sc(radius=5))
ring_sc_gc5

ring_sc_gc20 = add_gratings(ring_sc(radius=20))
ring_sc_gc20

print(ring_sc_gc5)

# ## Ipython
#
# This Jupyter Notebook uses an Interactive Python Terminal (Ipython). So you can interact with the code.
#
# For more details on Jupyter Notebooks, you can visit the [Jupyter website](https://jupyter.org/).
#
# The most common trick that you will see is that we use `?` to see the documentation of a function or `help(function)`

# +
# gf.components.coupler?
# -

help(gf.components.coupler)

# To see the source code of a function you can use `??`

# +
# gf.components.coupler??
# -

# To see which variables you have defined in the workspace you can type `whos`

# To time the execution time of a cell, you can add a `%time` on top of the cell

# +
# %time


def hi():
    print("hi")


hi()
# -

# For more Ipython tricks you can find many resources available online
