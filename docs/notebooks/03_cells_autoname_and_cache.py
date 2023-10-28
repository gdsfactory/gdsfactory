# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cell
#
# A `@cell` is a decorator for functions that return a Component. Make sure you add the `@cell` decorator to each function that returns a Component so you avoid having multiple components with the same name.
#
# Why do you need to add the `@cell` decorator?
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

# %%
import gdsfactory as gf


def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component("Unnamed_cells_can_cause_issues")
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend()
print(f"this cell {c.name!r} does NOT get automatic name")
c.plot()

# %%
mzi_with_bend_decorated = gf.cell(mzi_with_bend)
c = mzi_with_bend_decorated(radius=12)
print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()


# %%
@gf.cell
def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component()
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()


# %%
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

c = wg()
print(c)

print("second time you get this cell from the cache")
c = wg()
print(c)

print("If you call the cell with different parameters, the cell gets a different name")
c = wg(width=0.5)
print(c)


# %%
# Sometimes when you are changing the inside code of the function, you need to use `cache=False` to **ignore** the cache.

c = wg(cache=False)


# %% [markdown]
#
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
#

# %%
c = wg()

c.metadata["changed"]
c.metadata["default"]
c.metadata["full"]
c.pprint()


# %%
# thanks to `gf.cell` you can also add any metadata `info` relevant to the cell
c = wg(length=3, info=dict(polarization="te", wavelength=1.55))
c.pprint()
print(c.metadata["info"]["wavelength"])


# %% [markdown]
# ## Cache
#
# To avoid that 2 exact cells are not references of the same cell the `cell` decorator has a cache where if a component has already been built it will return the component from the cache

# %%


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


# %%

wg2 = wg()
# cell returns the same straight as before without having to run the function
print(wg2)  # notice that they have the same uuid (unique identifier)

wg2.plot()


# %%
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


# %% [markdown]
#
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
#   value is not a valid float
#
# ```
#
# by default `@cell` validates all arguments using [pydantic](https://pydantic-docs.helpmanual.io/usage/validation_decorator/#argument-types)
#
#


# %%
@gf.cell
def straigth_waveguide(length: float):
    print(type(length))
    return gf.components.straight(length=length)


# It will also convert an `int` to a `float`
c = straigth_waveguide(length=3)


# %% [markdown]
# # Create cells without `cell` decorator
#
# The cell decorator names cells deterministically and uniquely based on the name of the functions and its parameters.
#
# It also uses a caching mechanisms that improves performance and guards against duplicated names.
#
# The most common mistake new gdsfactory users make is to create cells without the `cell` decorator.
#
# ### Avoid naming cells manually: Use cell decorator
#
# Naming cells manually is susceptible to name collisions
#
# in GDS you can't have two cells with the same name.
#
# For example: this code will raise a `duplicated cell name ValueError`
#
# ```python
# import gdsfactory as gf
#
# c1 = gf.Component("wg")
# c1 << gf.components.straight(length=5)
#
#
# c2 = gf.Component("wg")
# c2 << gf.components.straight(length=50)
#
#
# c3 = gf.Component("waveguides")
# wg1 = c3 << c1
# wg2 = c3 << c2
# wg2.movey(10)
# c3
# ```
#
# **Solution**: Use the `gf.cell` decorator for automatic naming your components.

# %%
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()


@gf.cell
def wg(length: float = 3):
    return gf.components.straight(length=length)


print(wg(length=5))
print(wg(length=50))

# %% [markdown]
# ### Avoid Unnamed cells. Use `cell` decorator
#
# In the case of not wrapping the function with `cell` you will get unique names thanks to the unique identifier `uuid`.
#
# This name will be different and non-deterministic for different invocations of the script.
#
# However it will be hard for you to know where that cell came from.

# %%
c1 = gf.Component()
c2 = gf.Component()

print(c1.name)
print(c2.name)

# %% [markdown]
# Notice how gdsfactory raises a Warning when you save this `Unnamed` Components

# %%
c1.write_gds()


# %% [markdown]
# ### Avoid Intermediate Unnamed cells. Use `cell` decorator
#
# While creating a cell, you should not create intermediate cells, because they won't be Cached and you can end up with duplicated cell names or name conflicts, where one of the cells that has the same name as the other will be replaced.
#


# %%
@gf.cell
def die_bad():
    """c1 is an intermediate Unnamed cell"""
    c1 = gf.Component()
    _ = c1 << gf.components.straight(length=10)
    return gf.components.die_bbox(c1, street_width=10)


c = die_bad(cache=False)
print(c.references)
c.plot()

# %% [markdown]
# **Solution1** Don't use intermediate cells
#


# %%
@gf.cell
def die_good():
    c = gf.Component()
    _ = c << gf.components.straight(length=10)
    _ = c << gf.components.die_bbox_frame(c.bbox, street_width=10)
    return c


c = die_good(cache=False)
print(c.references)
c.plot()

# %% [markdown]
# **Solution2** You can flatten the cell, but you will lose the memory savings from cell references. Solution1 is more elegant.
#


# %%
@gf.cell
def die_flat():
    """c will be an intermediate unnamed cell"""
    c = gf.Component()
    _ = c << gf.components.straight(length=10)
    c2 = gf.components.die_bbox(c, street_width=10)
    c2 = c2.flatten()
    return c2


c = die_flat(cache=False)
print(c.references)
c.plot()

# %%
import gdsfactory as gf


@gf.cell
def dangerous_intermediate_cells(width=0.5):
    """Example that will show the dangers of using intermediate cells."""
    c = gf.Component("safe")

    c2 = gf.Component(
        "dangerous"
    )  # This should be forbidden as it will create duplicated cells
    _ = c2 << gf.components.hline(width=width)
    _ = c << c2

    return c


@gf.cell
def using_dangerous_intermediate_cells():
    """Example on how things can go wrong.

    Here we try to create to lines with different widths
    they end up with two duplicated cells and a name collision on the intermediate cell
    """
    c = gf.Component()
    _ = c << dangerous_intermediate_cells(width=0.5)
    r3 = c << dangerous_intermediate_cells(width=2)
    r3.movey(5)
    return c


c = using_dangerous_intermediate_cells()
c.plot_klayout()

# %%
for component in c.get_dependencies(recursive=True):
    if not component._locked:
        print(
            f"Component {component.name!r} was NOT properly locked. "
            "You need to write it into a function that has the @cell decorator."
        )

# %%
