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
# # Common mistakes
#
# ## 1. Creating cells without `cell` decorator
#
# The cell decorator names cells deterministically and uniquely based on the name of the functions and its parameters.
#
# It also uses a caching mechanisms that improves performance and guards against duplicated names.
#
# ### 1.1 naming cells manually
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

gf.config.rich_output()
PDK = gf.get_generic_pdk()
PDK.activate()


@gf.cell
def wg(length: float = 3):
    return gf.components.straight(length=length)


print(wg(length=5))
print(wg(length=50))

# %% [markdown]
# ### 1.2 Not naming components with a unique and deterministic name
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
# ### 1.3 Intermediate Unnamed cells
#
# While creating a cell, you should not create intermediate cells, because they won't be Cached and you can end up with duplicated cell names or name conflicts, where one of the cells that has the same name as the other will be replaced.
#


# %%
@gf.cell
def die_bad():
    """c1 is an intermediate Unnamed cell"""
    c1 = gf.Component()
    c1 << gf.components.straight(length=10)
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
    c << gf.components.straight(length=10)
    c << gf.components.die_bbox_frame(c.bbox, street_width=10)
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
    c << gf.components.straight(length=10)
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
    c2 << gf.components.hline(width=width)
    c << c2

    return c


@gf.cell
def using_dangerous_intermediate_cells():
    """Example on how things can go wrong.

    Here we try to create to lines with different widths
    they end up with two duplicated cells and a name collision on the intermediate cell
    """
    c = gf.Component()
    c << dangerous_intermediate_cells(width=0.5)
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
