# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Components with hierarchy
#
# ![](https://i.imgur.com/3pczkyM.png)
#
# You can define components Parametric cells (waveguides, bends, couplers) with basic input parameters (width, length, radius ...) and reuse the PCells in more complex PCells.

# +
from functools import partial
import toolz

from gdsfactory.typings import ComponentSpec

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()


# +
@gf.cell
def bend_with_straight(
    bend: ComponentSpec = gf.components.bend_euler,
    straight: ComponentSpec = gf.components.straight,
) -> gf.Component:
    c = gf.Component()
    b = gf.get_component(bend)
    s = gf.get_component(straight)

    bref = c << b
    sref = c << s

    sref.connect("o2", bref.ports["o2"])
    c.info["length"] = b.info["length"] + s.info["length"]
    return c


c = bend_with_straight()
print(c.metadata["info"]["length"])
c
# -

# ## ComponentSpec
#
# When defining a `Parametric cell` you can use other `ComponentSpec` as an argument. It can be a:
#
# 1. string: function name of a cell registered on the active PDK. `"bend_circular"`
# 2. dict: `dict(component='bend_circular', settings=dict(radius=20))`
# 3. function: Using `functools.partial` you can customize the default parameters of a function.

# ### 1. string

c = bend_with_straight(bend="bend_circular")
c

# ### 2. dict
# Lets **customize** the functions that we pass.
# For example, we want to increase the radius of the bend from the default 10um to 20um.

c = bend_with_straight(bend=dict(component="bend_circular", settings=dict(radius=20)))
c

# ### 3. function
#
# Partial lets you define different default parameters for a function, so you can modify the settings for the child cells.

c = bend_with_straight(bend=gf.partial(gf.components.bend_circular, radius=30))
c

bend20 = partial(gf.components.bend_circular, radius=20)
b = bend20()
b

type(bend20)

bend20.func.__name__

bend20.keywords

b = bend_with_straight(bend=bend20)
print(b.metadata["info"]["length"])
b

# You can still modify the bend to have any bend radius
b3 = bend20(radius=10)
b3

# ## PDK custom fab
#
# You can define a new PDK by creating function that customize partial parameters of the generic functions.
#
# Lets say that this PDK uses layer (41, 0) for the pads (instead of the layer used in the generic pad function).
#
# You can also access `functools.partial` from `gf.partial`

pad = gf.partial(gf.components.pad, layer=(41, 0))

c = pad()
c

# ## Composing functions
#
# You can combine more complex functions out of smaller functions.
#
# Lets say that we want to add tapers and grating couplers to a wide waveguide.

c1 = gf.components.straight()
c1

straight_wide = gf.partial(gf.components.straight, width=3)
c3 = straight_wide()
c3

c1 = gf.components.straight(width=3)
c1

c2 = gf.add_tapers(c1)
c2

c2.metadata_child["changed"]  # You can still access the child metadata

c3 = gf.routing.add_fiber_array(c2, with_loopback=False)
c3

c3.metadata_child["changed"]  # You can still access the child metadata

# Lets do it with a **single** step thanks to `toolz.pipe`

# +
add_fiber_array = gf.partial(gf.routing.add_fiber_array, with_loopback=False)
add_tapers = gf.add_tapers

# pipe is more readable than the equivalent add_fiber_array(add_tapers(c1))
c3 = toolz.pipe(c1, add_tapers, add_fiber_array)
c3
# -

# we can even combine `add_tapers` and `add_fiber_array` thanks to `toolz.compose` or `toolz.compose`
#
# For example:

add_tapers_fiber_array = toolz.compose_left(add_tapers, add_fiber_array)
c4 = add_tapers_fiber_array(c1)
c4

# is equivalent to

c5 = add_fiber_array(add_tapers(c1))
c5

# as well as equivalent to

add_tapers_fiber_array = toolz.compose(add_fiber_array, add_tapers)
c6 = add_tapers_fiber_array(c1)
c6

# or

c7 = toolz.pipe(c1, add_tapers, add_fiber_array)
c7

c7.metadata_child["changed"]  # You can still access the child metadata

c7.metadata["child"]["child"]["name"]

c7.metadata["child"]["child"]["function_name"]

c7.metadata["changed"].keys()
