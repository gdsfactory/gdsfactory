# # Cell
#
# A cell is a function that returns a Component.
#
# Make sure you add the `@cell` decorator to each function that returns a Component.
#
# `@cell` comes from PCell `parametric cell`, where the function returns a different Component depending on the input parameters.
#
# Why do we need cells?
#
# - In GDS each component must have a unique name. Ideally the name is also consistent from run to run, in case you want to merge GDS files that were created at different times or computers.
# - Two components stored in the GDS file cannot have the same name. They need to be references (instances) of the same component. See `References tutorial`. That way we only have to store the component in memory once and all the references are just pointers to that component.
#
# What does the `@cell` decorator does?
#
# 1. Gives the component a unique name depending on the parameters that you pass to it.
# 2. Creates a cache of components where we use the name as the key. The first time the function runs, the cache stores the component, so the second time, you get the component directly from the cache, so you don't create the same component twice.
#
#
# A decorator is a function that runs over a function, so when you do.
#
# ```python
# @gf.cell
# def mzi_with_bend():
#     c = gf.Component()
#     mzi = c << gf.components.mzi()
#     bend = c << gf.components.bend_euler()
#     return c
# ```
# it's equivalent to
#
# ```python
# def mzi_with_bend():
#     c = gf.Component()
#     mzi = c << gf.components.mzi()
#     bend = c << gf.components.bend_euler(radius=radius)
#     return c
#
#
# mzi_with_bend_decorated = gf.cell(mzi_with_bend)
# ```
#
# Lets see how it works.

# +
from gdsfactory.cell import print_cache
import gdsfactory as gf

gf.config.rich_output()

PDK = gf.get_generic_pdk()
PDK.activate()


def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component("Unnamed_cells_can_cause_issues")
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend()
print(f"this cell {c.name!r} does NOT get automatic name")
c.plot()
# -

mzi_with_bend_decorated = gf.cell(mzi_with_bend)
c = mzi_with_bend_decorated(radius=10)
print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()


# +
@gf.cell
def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component()
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()


# -


@gf.cell
def wg(length=10, width=1, layer=(1, 0)):
    print("BUILDING waveguide")
    c = gf.Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="o1", center=[0, width / 2], width=width, orientation=180, layer=layer
    )
    c.add_port(
        name="o2", center=[length, width / 2], width=width, orientation=0, layer=layer
    )
    return c


# See how the cells get the name from the parameters that you pass them

# +
c = wg()
print(c)

print("second time you get this cell from the cache")
c = wg()
print(c)

print("If you call the cell with different parameters, the cell gets a different name")
c = wg(width=0.5)
print(c)
# -

# Sometimes when you are changing the inside code of the function, you need to use `cache=False` to **ignore** the cache.

c = wg(cache=False)

# ## Metadata
#
# Together with the GDS file that you send to the foundry you can also store metadata in YAML for each cell containing all the settings that we used to build the GDS.
#
# the metadata will consists of all the parameters that were passed to the component function as well as derived properties
#
# - settings: includes all component metadata:
#     - changed: changed settings.
#     - child: child settings.
#     - default: includes the default cell function settings.
#     - full: full settings.
#     - function_name: from the cell function.
#     - info: metadata in Component.info dict.
#     - module: python module where you can find the cell function.
#     - name: for the component
# - ports: port name, width, orientation

c = wg()

c.metadata["changed"]

c.metadata["default"]

c.metadata["full"]

c.pprint()

# thanks to `gf.cell` you can also add any metadata `info` relevant to the cell

c = wg(length=3, info=dict(polarization="te", wavelength=1.55))

c.pprint()

print(c.metadata["info"]["wavelength"])


# ## Cache
#
# To avoid that 2 exact cells are not references of the same cell the `cell` decorator has a
# cache where if a component has already been built it will return the component
# from the cache
#


@gf.cell
def wg(length=10, width=1):
    c = gf.Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=(1, 0))
    print("BUILDING waveguide")
    return c


# +
gf.clear_cache()

wg1 = wg()  # cell builds a straight
print(wg1)
# -

wg2 = wg()
# cell returns the same straight as before without having to run the function
print(wg2)  # notice that they have the same uuid (unique identifier)

wg2

# Lets say that you change the code of the straight function in a Jupyter Notebook like this one.  (I mostly use Vim/VsCode/Pycharm for creating new cells in python)

print_cache()

wg3 = wg()
wg4 = wg(length=11)

print_cache()

gf.clear_cache()

# To enable nice notebook tutorials, every time we show a cell in Matplotlib or Klayout, you can clear the cache,
#
# in case you want to develop cells in Jupyter Notebooks or an IPython kernel

print_cache()  # cache is now empty

# ## Validate argument types
#
# By default, also `@cell` validates arguments based on their type annotations.
# To make sure you pass the correct arguments to the cell function it runs a validator that checks the type annotations for the function.
#
#
# For example this will be correct
#
# ```python
# import gdsfactory as gf
#
#
# @gf.cell
# def straigth_waveguide(length: float):
#     return gf.components.straight(length=length)
#
#
# component = straigth_waveguide(length=3)
# ```
#
# While this will raise an error, because you are passing a length that is a string, so it cannot convert it to a float
#
#
# ```python
# component = straigth_waveguide(length="long")
# ```
#
# ```bash
# ValidationError: 1 validation error for StraigthWaveguide
# length
#   value is not a valid float (type=type_error.float)
#
# ```
#
# by default `@cell` validates all arguments using [pydantic](https://pydantic-docs.helpmanual.io/usage/validation_decorator/#argument-types)


# +
@gf.cell
def straigth_waveguide(length: float):
    print(type(length))
    return gf.components.straight(length=length)


# It will also convert an `int` to a `float`
c = straigth_waveguide(length=3)
