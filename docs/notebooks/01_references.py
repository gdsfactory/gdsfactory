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
# # References and ports
#
# gdsfactory defines your component once in memory and can add multiple References (Instances) to the same component.

# %% [markdown]
# As you build components you can include references to other components. Adding a reference is like having a pointer to a component.
#
# The GDSII specification allows the use of references, and similarly gdsfactory uses them (with the `add_ref()` function).
# what is a reference? Simply put:  **A reference does not contain any geometry. It only *points* to an existing geometry**.
#
# Say you have a ridiculously large polygon with 100 billion vertices that you call BigPolygon. It's huge, and you need to use it in your design 250 times.
# Well, a single copy of BigPolygon takes up 1MB of memory, so you don't want to make 250 copies of it
# You can instead *references* the polygon 250 times.
# Each reference only uses a few bytes of memory -- it only needs to know the memory address of BigPolygon, position, rotation and mirror.
# This way, you can keep one copy of BigPolygon and use it again and again.
#
# You can start by making a blank `Component` and add a single polygon to it.

# %%
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# Create a blank Component
p = gf.Component("component_with_polygon")

# Add a polygon
xpts = [0, 0, 5, 6, 9, 12]
ypts = [0, 1, 1, 2, 2, 0]
p.add_polygon([xpts, ypts], layer=(2, 0))

# plot the Component with the polygon in it
p.plot()

# %% [markdown]
# Now, you want to reuse this polygon repeatedly without creating multiple copies of it.
#
# To do so, you need to make a second blank `Component`, this time called `c`.
#
# In this new Component you *reference* our Component `p` which contains our polygon.

# %%
c = gf.Component("Component_with_references")  # Create a new blank Component
poly_ref = c.add_ref(p)  # Reference the Component "p" that has the polygon in it
c.plot()

# %% [markdown]
# you just made a copy of your polygon -- but remember, you didn't actually
# make a second polygon, you just made a reference (aka pointer) to the original
# polygon.  Let's add two more references to `c`:

# %%
poly_ref2 = c.add_ref(p)  # Reference the Component "p" that has the polygon in it
poly_ref3 = c.add_ref(p)  # Reference the Component "p" that has the polygon in it
c.plot()

# %% [markdown]
# Now you have 3x polygons all on top of each other.  Again, this would appear
# useless, except that you can manipulate each reference independently. Notice that
# when you called `c.add_ref(p)` above, we saved the result to a new variable each
# time (`poly_ref`, `poly_ref2`, and `poly_ref3`)?  You can use those variables to
# reposition the references.

# %%
poly_ref2.rotate(90)  # Rotate the 2nd reference we made 90 degrees
poly_ref3.rotate(180)  # Rotate the 3rd reference we made 180 degrees
c.plot()

# %% [markdown]
# Now you're getting somewhere! You've only had to make the polygon once, but you're
# able to reuse it as many times as you want.
#
# ## Modifying the referenced geometry
#
# What happens when you change the original geometry that the reference points to?  In your case, your references in
# `c` all point to the Component `p` that with the original polygon.  Let's try
# adding a second polygon to `p`.
#
# First you add the second polygon and make sure `P` looks like you expect:

# %%
# Add a 2nd polygon to "p"
xpts = [14, 14, 16, 16]
ypts = [0, 2, 2, 0]
p.add_polygon([xpts, ypts], layer=(1, 0))
p

# %% [markdown]
# That looks good.  Now let's find out what happened to `c` that contains the
# three references.  Keep in mind that you have not modified `c` or executed any
# functions/operations on `c` -- all you have done is modify `p`.

# %%
c.plot()

# %% [markdown]
#  **When you modify the original geometry, all of the
# references automatically reflect the modifications.**  This is very powerful,
# because you can use this to make very complicated designs from relatively simple
# elements in a computation- and memory-efficient way.
#
# Let's try making references a level deeper by referencing `c`.  Note here we use
# the `<<` operator to add the references -- this is just shorthand, and is
# exactly equivalent to using `add_ref()`

# %%
c2 = gf.Component("array_sample")  # Create a new blank Component
d_ref1 = c2.add_ref(c)  # Reference the Component "c" that 3 references in it
d_ref2 = c2 << c  # Use the "<<" operator to create a 2nd reference to c.plot()
d_ref3 = c2 << c  # Use the "<<" operator to create a 3rd reference to c.plot()

d_ref1.move([20, 0])
d_ref2.move([40, 0])

c2

# %% [markdown]
# As you've seen you have two ways to add a reference to our component:
#
# 1. create the reference and add it to the component

# %%
c = gf.Component("reference_sample")
w = gf.components.straight(width=0.6)
wr = w.ref()
c.add(wr)
c.plot()

# %% [markdown]
# 2. or do it in a single line

# %%
c = gf.Component("reference_sample_shorter_syntax")
wr = c << gf.components.straight(width=0.6)
c.plot()

# %% [markdown]
# in both cases you can move the reference `wr` after created

# %%
c = gf.Component("two_references")
wr1 = c << gf.components.straight(width=0.6)
wr2 = c << gf.components.straight(width=0.6)
wr2.movey(10)
c.add_ports(wr1.get_ports_list(), prefix="bot_")
c.add_ports(wr2.get_ports_list(), prefix="top_")

# %%
c.ports

# %% [markdown]
# You can also auto_rename ports using gdsfactory default convention, where ports are numbered clockwise starting from the bottom left

# %%
c.auto_rename_ports()

# %%
c.ports

# %%
c.plot()

# %% [markdown]
# ## Arrays of references
#
# In GDS, there's a type of structure called a "ComponentReference" which takes a cell and repeats it NxM times on a fixed grid spacing. For convenience, `Component` includes this functionality with the add_array() function.
# Note that CellArrays are not compatible with ports (since there is no way to access/modify individual elements in a GDS cellarray)
#
# gdsfactory also provides with more flexible arrangement options if desired, see for example `grid()` and `packer()`.
#
# As well as `gf.components.array`
#
# Let's make a new Component and put a big array of our Component `c` in it:

# %%
c3 = gf.Component("array_of_references")  # Create a new blank Component
aref = c3.add_array(
    c, columns=6, rows=3, spacing=[20, 15]
)  # Reference the Component "c" 3 references in it with a 3 rows, 6 columns array
c3

# %% [markdown]
# CellArrays don't have ports and there is no way to access/modify individual elements in a GDS cellarray.
#
# gdsfactory provides you with similar functions in `gf.components.array` and `gf.components.array_2d`

# %%
c4 = gf.Component("demo_array")  # Create a new blank Component
aref = c4 << gf.components.array(component=c, columns=3, rows=2)
c4.add_ports(aref.get_ports_list())
c4


# %%
help(gf.components.array)


# %% [markdown]
# You can also create an array of references for periodic structures. Let's create a [Distributed Bragg Reflector](https://picwriter.readthedocs.io/en/latest/components/dbr.html)
#


# %%
@gf.cell
def dbr_period(w1=0.5, w2=0.6, l1=0.2, l2=0.4, straight=gf.components.straight):
    """Return one DBR period."""
    c = gf.Component()
    r1 = c << straight(length=l1, width=w1)
    r2 = c << straight(length=l2, width=w2)
    r2.connect(port="o1", destination=r1.ports["o2"])
    c.add_port("o1", port=r1.ports["o1"])
    c.add_port("o2", port=r2.ports["o2"])
    return c


l1 = 0.2
l2 = 0.4
n = 3
period = dbr_period(l1=l1, l2=l2)
period

# %%
dbr = gf.Component("DBR")
dbr.add_array(period, columns=n, rows=1, spacing=(l1 + l2, 100))
dbr

# %% [markdown]
# Finally we need to add ports to the new component

# %%
p0 = dbr.add_port("o1", port=period.ports["o1"])
p1 = dbr.add_port("o2", port=period.ports["o2"])

p1.center = [(l1 + l2) * n, 0]
dbr

# %% [markdown]
# ## Connect references
#
# We have seen that once you create a reference you can manipulate the reference to move it to a location. Here we are going to connect that reference to a port. Remember that we follow that a certain reference `source` connects to a `destination` port

# %%
bend = gf.components.bend_circular()
bend

# %%
c = gf.Component("sample_reference_connect")

mmi = c << gf.components.mmi1x2()
b = c << gf.components.bend_circular()
b.connect("o1", destination=mmi.ports["o2"])

c.add_port("o1", port=mmi.ports["o1"])
c.add_port("o2", port=b.ports["o2"])
c.add_port("o3", port=mmi.ports["o3"])
c.plot()

# %% [markdown]
# You can also access the ports as `reference[port_name]` instead of `reference.ports[port_name]`

# %%
c = gf.Component("sample_reference_connect_simpler")

mmi = c << gf.components.mmi1x2()
b = c << gf.components.bend_circular()
b.connect("o1", destination=mmi["o2"])

c.add_port("o1", port=mmi["o1"])
c.add_port("o2", port=b["o2"])
c.add_port("o3", port=mmi["o3"])
c.plot()

# %% [markdown]
# Notice that `connect` mates two ports together and does not imply that ports will remain connected.
#

# %% [markdown]
# ## Port
#
# You can name the ports as you want and use `gf.port.auto_rename_ports(prefix='o')` to rename them later on.
#
# Here is the default naming convention.
#
# Ports are numbered clock-wise starting from the bottom left corner.
#
# Optical ports have `o` prefix and Electrical ports `e` prefix.
#
# The port naming comes in most cases from the `gdsfactory.cross_section`. For example:
#
# - `gdsfactory.cross_section.strip`  has ports `o1` for input and `o2` for output.
# - `gdsfactory.cross_section.metal1` has ports `e1` for input and `e2` for output.

# %%
size = 4
c = gf.components.nxn(west=2, south=2, north=2, east=2, xsize=size, ysize=size)
c.plot()

# %%
c = gf.components.straight_heater_metal(length=30)
c.plot()

# %%
c.ports

# %% [markdown]
# You can get the optical ports by `layer`

# %%
c.get_ports_dict(layer=(1, 0))

# %% [markdown]
# or by `width`

# %%
c.get_ports_dict(width=0.5)

# %%
c0 = gf.components.straight_heater_metal()
c0.ports

# %%
c1 = c0.copy()
c1.auto_rename_ports_layer_orientation()
c1.ports

# %%
c2 = c0.copy()
c2.auto_rename_ports()
c2.ports

# %% [markdown]
# You can also rename them with a different port naming convention
#
# - prefix: add `e` for electrical `o` for optical
# - clockwise
# - counter-clockwise
# - orientation `E` East, `W` West, `N` North, `S` South
#
#
# Here is the default one we use (clockwise starting from bottom left west facing port)
#
# ```
#              3   4
#              |___|_
#          2 -|      |- 5
#             |      |
#          1 -|______|- 6
#              |   |
#              8   7
#
# ```

# %%
c = gf.Component("demo_ports")
nxn = gf.components.nxn(west=2, north=2, east=2, south=2, xsize=4, ysize=4)
ref = c.add_ref(nxn)
c.add_ports(ref.ports)
c.plot()

# %%
ref.get_ports_list()  # by default returns ports clockwise starting from bottom left west facing port

# %%
c.auto_rename_ports()
c.plot()

# %% [markdown]
# You can also get the ports counter-clockwise
#
# ```
#              4   3
#              |___|_
#          5 -|      |- 2
#             |      |
#          6 -|______|- 1
#              |   |
#              7   8
#
# ```

# %%
c.auto_rename_ports_counter_clockwise()
c.plot()

# %%
c.get_ports_list(clockwise=False)

# %%
c.ports_layer

# %%
c.port_by_orientation_cw("W0")

# %%
c.port_by_orientation_ccw("W1")

# %% [markdown]
# Lets extend the East facing ports (orientation = 0 deg)

# %%
cross_section = gf.cross_section.strip()

nxn = gf.components.nxn(
    west=2, north=2, east=2, south=2, xsize=4, ysize=4, cross_section=cross_section
)
c = gf.components.extension.extend_ports(component=nxn, orientation=0)
c.plot()

# %%
c.ports

# %%
df = c.get_ports_pandas()
df

# %%
df[df.port_type == "optical"]

# %% [markdown]
# ## Port markers (Pins)
#
# You can add pins (port markers) to each port. Different foundries do this differently, so gdsfactory supports all of them.
#
# - square with port inside the component.
# - square centered (half inside, half outside component).
# - triangular pointing towards the outside of the port.
# - path (SiEPIC).
#
#
# by default Component.show() will add triangular pins, so you can see the direction of the port in Klayout.

# %%
gf.components.mmi1x2(decorator=gf.add_pins.add_pins)

# %%
gf.components.mmi1x2(decorator=gf.add_pins.add_pins_triangle)

# %% [markdown]
# ## Component_sequence
#
# When you have repetitive connections you can describe the connectivity as an ASCII map

# %%
bend180 = gf.components.bend_circular180()
wg_pin = gf.components.straight_pin(length=40)
wg = gf.components.straight()

# Define a map between symbols and (component, input port, output port)
symbol_to_component = {
    "D": (bend180, "o1", "o2"),
    "C": (bend180, "o2", "o1"),
    "P": (wg_pin, "o1", "o2"),
    "-": (wg, "o1", "o2"),
}

# Generate a sequence
# This is simply a chain of characters. Each of them represents a component
# with a given input and and a given output

sequence = "DC-P-P-P-P-CD"
component = gf.components.component_sequence(
    sequence=sequence, symbol_to_component=symbol_to_component
)
component.name = "component_sequence"
component.plot()

# %% [markdown]
# As the sequence is defined as a string you can use the string operations to easily build complex sequences

# %% [markdown]
# ## Movement
#
# You can move, rotate and mirror ComponentReference as well as `Port`, `Polygon`, `ComponentReference`, `Label`, and `Group`

# +
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()

PDK = get_generic_pdk()

PDK.activate()

# Start with a blank Component
c = gf.Component("demo_movement")

# Create some more Components with shapes
T = gf.components.text("hello", size=10, layer=(1, 0))
E = gf.components.ellipse(radii=(10, 5), layer=(2, 0))
R = gf.components.rectangle(size=(10, 3), layer=(3, 0))

# Add the shapes to c as references
text = c << T
ellipse = c << E
rect1 = c << R
rect2 = c << R

c.plot()
# -

c = gf.Component("move_one_ellipse")
e1 = c << gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e2 = c << gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e1.movex(10)
c.plot()

c = gf.Component("move_one_ellipse_xmin")
e1 = c << gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e2 = c << gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e2.xmin = e1.xmax
c.plot()

# Now you can practice move and rotate the objects.

c = gf.Component("two_ellipses_on_top_of_each_other")
E = gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e1 = c << E
e2 = c << E
c.plot()

c = gf.Component("ellipse_moved")
e = gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e1 = c << e
e2 = c << e
e2.move(origin=[5, 5], destination=[10, 10])  # Translate by dx = 5, dy = 5
c.plot()

c = gf.Component("ellipse_moved_v2")
e = gf.components.ellipse(radii=(10, 5), layer=(2, 0))
e1 = c << e
e2 = c << e
e2.move([5, 5])  # Translate by dx = 5, dy = 5
c.plot()

# +
c = gf.Component("rectangles")
r = gf.components.rectangle(size=(10, 5), layer=(2, 0))
rect1 = c << r
rect2 = c << r

rect1.rotate(45)  # Rotate the first straight by 45 degrees around (0,0)
rect2.rotate(
    -30, center=[1, 1]
)  # Rotate the second straight by -30 degrees around (1,1)
c.plot()
# -

c = gf.Component("mirror_demo")
text = c << gf.components.text("hello")
text.mirror(p1=[1, 1], p2=[1, 3])  # Reflects across the line formed by p1 and p2
c.plot()

c = gf.Component("hello")
text = c << gf.components.text("hello")
c.plot()

# Each Component and ComponentReference object has several properties which can be
# used
# to learn information about the object (for instance where it's center coordinate
# is).  Several of these properties can actually be used to move the geometry by
# assigning them new values.
#
# Available properties are:
#
# - `xmin` / `xmax`: minimum and maximum x-values of all points within the object
# - `ymin` / `ymax`: minimum and maximum y-values of all points within the object
# - `x`: centerpoint between minimum and maximum x-values of all points within the
# object
# - `y`: centerpoint between minimum and maximum y-values of all points within the
# object
# - `bbox`: bounding box (see note below) in format ((xmin,ymin),(xmax,ymax))
# - `center`: center of bounding box

print("bounding box:")
print(
    text.bbox
)  # Will print the bounding box of text in terms of [(xmin, ymin), (xmax, ymax)]
print("xsize and ysize:")
print(text.xsize)  # Will print the width of text in the x dimension
print(text.ysize)  # Will print the height of text in the y dimension
print("center:")
print(text.center)  # Gives you the center coordinate of its bounding box
print("xmax")
print(text.xmax)  # Gives you the rightmost (+x) edge of the text bounding box

# Let's use these properties to manipulate our shapes to arrange them a little
# better

# +
c = gf.Component("canvas")
text = c << gf.components.text("hello")
E = gf.components.ellipse(radii=(10, 5), layer=(3, 0))
R = gf.components.rectangle(size=(10, 5), layer=(2, 0))
rect1 = c << R
rect2 = c << R
ellipse = c << E

c.plot()

# +
# First let's center the ellipse
ellipse.center = [
    0,
    0,
]  # Move the ellipse such that the bounding box center is at (0,0)

# Next, let's move the text to the left edge of the ellipse
text.y = (
    ellipse.y
)  # Move the text so that its y-center is equal to the y-center of the ellipse
text.xmax = ellipse.xmin  # Moves the ellipse so its xmax == the ellipse's xmin

# Align the right edge of the rectangles with the x=0 axis
rect1.xmax = 0
rect2.xmax = 0

# Move the rectangles above and below the ellipse
rect1.ymin = ellipse.ymax + 5
rect2.ymax = ellipse.ymin - 5

c.plot()
# -

# In addition to working with the properties of the references inside the
# Component,
# we can also manipulate the whole Component if we want.  Let's try mirroring the
# whole Component `c`:

# +
print(c.xmax)  # Prints out '10.0'

c2 = c.mirror((0, 1))  # Mirror across line made by (0,0) and (0,1)
c2.plot()
# -

# A bounding box is the smallest enclosing box which contains all points of the geometry.

c = gf.Component("hi_bbox")
text = c << gf.components.text("hi")
bbox = text.bbox
c << gf.components.bbox(bbox=bbox, layer=(2, 0))
c.plot()

# gf.get_padding_points can also add a bbox with respect to the bounding box edges
c = gf.Component("sample_padding")
text = c << gf.components.text("bye")
device_bbox = text.bbox
c.add_polygon(gf.get_padding_points(text, default=1), layer=(2, 0))
c.plot()

# When we query the properties of c, they will be calculated with respect to this
# bounding-rectangle.  For instance:

# +
print("Center of Component c:")
print(c.center)

print("X-max of Component c:")
print(c.xmax)
# -

c = gf.Component("rect")
R = gf.components.rectangle(size=(10, 3), layer=(2, 0))
rect1 = c << R
c.plot()

# You can chain many of the movement/manipulation functions because they all return the object they manipulate.
#
# For instance you can combine two expressions:

rect1.rotate(angle=37)
rect1.move([10, 20])
c.plot()

# ...into this single-line expression

c = gf.Component("single_expression")
R = gf.components.rectangle(size=(10, 3), layer=(2, 0))
rect1 = c << R
rect1.rotate(angle=37).move([10, 20])
c.plot()
