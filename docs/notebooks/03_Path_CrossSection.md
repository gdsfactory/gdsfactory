# Path and CrossSection

You can create a `Path` in gdsfactory and extrude it with an arbitrary `CrossSection`.

Let us create a path:

- Create a blank `Path`.
- Append points to the `Path` by either using the built-in functions (`arc()`, `straight()`, `euler()` ...) or by providing your own lists of points.
- Specify `CrossSection` with layers and offsets.
- Extrude `Path` with a `CrossSection` to create a Component with the path polygons in it.

```python
import matplotlib.pyplot as plt
import numpy as np

import gdsfactory as gf

gf.gpdk.PDK.activate()

```

## Path

The first step is to generate the list of points we want the path to follow.
Let us now start out by creating a blank `Path` and using the built-in functions to
make a few smooth turns.

```python
p1 = gf.path.straight(length=5)

# This creates a curved path segment using an Euler bend profile,
# which is a curve with a continuously changing radius designed to minimize light loss. This specific bend turns by 45 degrees.
# By setting use_eff=False, you are telling the function to ignore complex calculations and instead create a simpler bend with a constant, user-specified radius.
p2 = gf.path.euler(radius=5, angle=45, p=0.5, use_eff=False)

# The + operator is used to concatenate the two paths.
# It takes the second path (p2) and appends it to the end of the first path (p1), ensuring a smooth, continuous transition.
p = p1 + p2
f = p.plot()
```

```python
p1 = gf.path.straight(length=5)
p2 = gf.path.euler(radius=5, angle=45, p=0.5, use_eff=False)
p = p2 + p1
f = p.plot()
```

```python
# Note: -angle rotations correspond to a clockwise turn.
P = gf.Path()
P += gf.path.arc(radius=10, angle=90)  # Circular arc.
P += gf.path.straight(length=10)  # Straight section.
P += gf.path.euler(radius=3, angle=-90)  # Euler bend (aka "racetrack" curve).
P += gf.path.straight(length=40)
P += gf.path.arc(radius=8, angle=-45)
P += gf.path.straight(length=10)
P += gf.path.arc(radius=8, angle=45)
P += gf.path.straight(length=10)

f = P.plot()
```

```python
p2 = P.copy().rotate(45)
f = p2.plot()
```

```python
P.points - p2.points
```

You can also modify our path in the same ways as any other gdsfactory object:

- Manipulation with `move()`, `rotate()`, `mirror()`, etc
- Accessing properties like `xmin`, `y`, `center`, `bbox`, etc

```python
P.movey(10)
P.xmin = 20
f = P.plot()
```

You can also check the length of the curve with the `length()` method:

```python
P.length()
```

<!-- #region -->
## CrossSection

Now that you have got your path defined, the next step is to define the cross-section of the path. To do this, you can create a blank `CrossSection` and add whatever cross-sections you want to it.
You can then combine the `Path` and the `CrossSection` using the `gf.path.extrude()` function to generate a component:


### Option 1: Single layer and width cross-section

The simplest option is to just set the cross-section to be a constant width by passing a number to `extrude()` like so:
<!-- #endregion -->

```python
# Extrude the Path and the cross-section.
# The extrude function converts a 1D path into a 2D shape by giving it a specified width.

c = gf.path.extrude(P, layer=(1, 0), width=1.5)
c.plot()
```

### Option 2: Arbitrary Cross-section

You can also extrude an arbitrary cross_section.


Now, what if we want a more complicated straight?  For instance, in some
photonic applications it is helpful to have a shallow etch that appears on either
side of the straight (often called a trench or sleeve).  Additionally, it might be nice
to have a port on either end of the center section so we can snap other
geometries to it.  Let us try adding something like that in:

```python
p = gf.path.straight()

# The code first defines three gf.Section objects, each representing a part of the total cross-section.
# s0: The central core section. It is 1 µm wide, centered at an offset of 0, and is on layer (1, 0).
# s1: A side section. It is 2 µm wide, its center is offset by +2 µm from the main centerline, and it is on layer (2, 0).
# s2: Another side section, identical to s1 but offset by -2 µm.
s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
s1 = gf.Section(width=2, offset=2, layer=(2, 0))
s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
x = gf.CrossSection(sections=(s0, s1, s2))

c = gf.path.extrude(p, cross_section=x)
c.draw_ports()
c.plot()
```

If you add more ports to a cross-section it also exposes its ports.

```python
p = gf.path.straight()

# Add a few "sections" to the cross-section.
s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
s1 = gf.Section(width=2, offset=2, layer=(2, 0), port_names=("e1", "e2"))
s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
x = gf.CrossSection(sections=(s0, s1, s2))

c = gf.path.extrude(p, cross_section=x)
c.draw_ports()
c.plot()
```

```python
p = gf.path.arc() # A 1D path in the shape of a 90-degree circular arc is created. This defines the centerline for the extrusion.

# Combine the Path and the cross-section.
b = gf.path.extrude(p, cross_section=x)
b.plot()
```

⚠️ Warning! for GS routing. You need to add a centered port for routing to work correctly.

```python
p = gf.path.straight()

# Add GS routing sections.
# GS routing is a specific auto-routing algorithm used in gdsfactory to create smooth, low-loss waveguide connections between component ports.
s0 = gf.Section(width=2, offset=0, layer=(2, 0), port_names=("g1", "g2"))
s1 = gf.Section(width=2, offset=4, layer=(2, 0))
x = gf.CrossSection(sections=(s0, s1), radius=8)
c = gf.path.extrude(p, cross_section=x)
pad = c
c_copy = c.copy()
c_copy.draw_ports()
c_copy.plot()
```

⚠️ Warning! for GS routing. You need to add a centered port for routing to work correctly.

```python
# Do not do this for GS routing. See solution below.
c2 = gf.Component()
pad1 = c2 << pad
pad2 = c2 << pad
pad2.move((100, 100))


# The gf.routing.route_bundle function is a powerful auto-router for creating multiple, parallel waveguide connections.
# [pad1.ports["g2"]]: A list of the starting ports for the routes.
# [pad2.ports["g1"]]: A list of the ending ports for the routes.
# sort_ports=True: An option that helps the router find the optimal, non-crossing paths when routing multiple waveguides.
# bend='bend_euler': Specifies that any curves in the route should be smooth Euler bends.
gf.routing.route_bundle(c2, [pad1.ports["g2"]], [pad2.ports["g1"]], cross_section=x, sort_ports=True, bend='bend_euler')
c2.plot()
```

For GSG routing it works well because the port is at the center. GSG routing is a specific auto-routing algorithm used in gdsfactory to create smooth, S-bend-like waveguide connections between component ports.

```python
p = gf.path.straight()

# Add a few "sections" to the cross-section
g = gf.Section(width=2, offset=0, layer=(2, 0), port_names=("e1", "e2"), port_types=('electrical', 'electrical'))
s0 = gf.Section(width=2, offset=-4, layer=(2, 0))
s1 = gf.Section(width=2, offset=4, layer=(2, 0))
x = gf.CrossSection(sections=(g, s0, s1), radius=8)
c = gf.path.extrude(p, cross_section=x)
c_copy = c.copy()
c_copy.draw_ports()
c_copy.plot()
```

```python
c2 = gf.Component()
pad1 = c2 << c
pad2 = c2 << c
pad2.move((100, 100))
gf.routing.route_bundle(c2, [pad1.ports["e2"]], [pad2.ports["e1"]], cross_section=x, port_type='electrical')
c2.plot()
```

For GS routing the recommended solution is adding a dummy / abstract layer in the middle, where we add the ports.

```python
p = gf.path.straight()

# Add GS routing sections.
# 99, 0 is an abstract layer that can be used to add ports to the path.
s0 = gf.Section(width=2, offset=0, layer=(99, 0), port_names=("e1", "e2"))
s1 = gf.Section(width=2, offset=-4, layer=(2, 0))
s2 = gf.Section(width=2, offset=+4, layer=(2, 0))
x = gf.CrossSection(sections=(s0, s1, s2), radius=8)
c = gf.path.extrude(p, cross_section=x)
pad = c
c_copy = c.copy()
c_copy.draw_ports()
c_copy.plot()
```

```python
c2 = gf.Component()
pad1 = c2 << c
pad2 = c2 << c
pad2.move((100, 100))
gf.routing.route_bundle(c2,
    [pad1.ports["e2"]],
    [pad2.ports["e1"]],
    cross_section=x,
    port_type='optical',
    bend='bend_euler',
    raise_on_error=True,
)
c2.plot()
```

### Option 3: Cross-section with ComponentAlongPath

You can also place components along a path, which is useful for wiring vias. A via is a vertical electrical connection that goes through the insulating layers of an integrated circuit to connect different layers of horizontal metal wiring.

```python
import gdsfactory as gf
from gdsfactory.cross_section import ComponentAlongPath

# Create the path.
p = gf.path.straight()
p += gf.path.arc(10)
p += gf.path.straight()

# Define a cross-section containing a via.
via = ComponentAlongPath(
    component=gf.c.rectangle(size=(1, 1), centered=True), spacing=5, padding=2
)
s = gf.Section(width=0.5, offset=0, layer=(1, 0), port_names=("in", "out"))
x = gf.CrossSection(sections=(s,), components_along_path=(via,))

# Combine the path with the cross-section.
c = gf.path.extrude(p, cross_section=x)
c.plot()
```

```python
import gdsfactory as gf
from gdsfactory.cross_section import ComponentAlongPath

# Create the path.
p = gf.path.straight()
p += gf.path.arc(10)
p += gf.path.straight()

# Define a cross-section with a via.
via0 = ComponentAlongPath(component=gf.c.via1(), spacing=5, padding=2, offset=0)
viap = ComponentAlongPath(component=gf.c.via1(), spacing=5, padding=2, offset=+2)
vian = ComponentAlongPath(component=gf.c.via1(), spacing=5, padding=2, offset=-2)
x = gf.CrossSection(sections=[s], components_along_path=(via0, viap, vian))

# Combine the path with the cross-section.
c = gf.path.extrude(p, cross_section=x)
c.plot()
```

## Path

You can pass `append()` lists of path segments.  This makes it easy to combine paths very quickly.
Below we show 3 examples using this functionality:

**Example 1:** Assemble a complex path by making a list of paths and passing it to `append()`.

```python
import gdsfactory as gf

P = gf.Path()

# Create the basic Path components.
left_turn = gf.path.euler(radius=4, angle=90)
right_turn = gf.path.euler(radius=4, angle=-90)
straight = gf.path.straight(length=10)

# Assemble a complex path by making a list of paths and passing it to `append()`.
# .append([...]): This method takes a list of path factories and adds them sequentially to the end of the existing path P.
# Each new segment starts where the previous one ended, creating a single, continuous path.
P.append(
    [
        straight,
        left_turn,
        straight,
        right_turn,
        straight,
        straight,
        right_turn,
        left_turn,
        straight,
    ]
)

f = P.plot()
```

```python
P = (
    straight
    + left_turn
    + straight
    + right_turn
    + straight
    + straight
    + right_turn
    + left_turn
    + straight
)
f = P.plot()
```

**Example 2:** Create an "S-turn" just by making a list of `[left_turn,
right_turn]`.

```python
P = gf.Path()

# Create an "S-turn" by making a list.
s_turn = [left_turn, right_turn]

P.append(s_turn)
f = P.plot()
```

**Example 3:** Repeat the S-turn 3 times by nesting our S-turn list in another list. Nesting means placing one data structure inside another of the same type. In this context, it means creating a "list of lists."

```python
P = gf.Path()

# Create an "S-turn" using a list.
s_turn = [left_turn, right_turn]

# Repeat the S-turn 3 times by nesting our S-turn list 3x times in another list.
triple_s_turn = [s_turn, s_turn, s_turn]

P.append(triple_s_turn)
f = P.plot()
```

Note you can also use the Path() constructor to immediately construct your Path:

```python
P = gf.Path([straight, left_turn, straight, right_turn, straight])
f = P.plot()
```

## Waypoint smooth paths

You can also build smooth paths between waypoints with the `smooth()` function.

```python
points = np.array([(20, 10), (40, 10), (20, 40), (50, 40), (50, 20), (70, 20)])
plt.plot(points[:, 0], points[:, 1], ".-")

# This functionensures that one unit on the x-axis is the same length as one unit on the y-axis.
# This is crucial for plots where the geometric shape is important.
plt.axis("equal")
```

```python
points = np.array([(20, 10), (40, 10), (20, 40), (50, 40), (50, 20), (70, 20)])

P = gf.path.smooth(
    points=points,
    radius=2,
    bend=gf.path.euler,  # Alternatively, use pp.arc, which will create a constant-radius bend.
    use_eff=False,
)
f = P.plot()
```

## Waypoint sharp paths

It is also possible to make more traditional angular paths (e.g. electrical wires) in a few different ways.

**Example 1:** Using a simple list of points:

```python
P = gf.Path([(20, 10), (30, 10), (40, 30), (50, 30), (50, 20), (70, 20)])
f = P.plot()
```

**Example 2:** Using the "turn and move" method, where you manipulate the end angle of the path so that when you append points to it they are in the correct direction.  *Note: It is crucial that the number of points per straight section is set to 2 (`gf.path.straight(length, num_pts = 2)`) otherwise the extrusion algorithm will show defects.*

```python
P = gf.Path()
P += gf.path.straight(length=10, npoints=2)
P.end_angle += 90  # "Turn" 90 deg (left).
P += gf.path.straight(length=10, npoints=2)  # "Walk" length of 10.
P.end_angle += -135  # "Turn" -135 degrees (right).
P += gf.path.straight(length=15, npoints=2)  # "Walk" length of 15.
P.end_angle = 0  # Force the direction to be 0 degrees.
P += gf.path.straight(length=10, npoints=2) 
f = P.plot()
```

```python
s0 = gf.Section(width=1, offset=0, layer=(1, 0))
s1 = gf.Section(width=1.5, offset=2.5, layer=(2, 0))
s2 = gf.Section(width=1.5, offset=-2.5, layer=(3, 0))
X = gf.CrossSection(sections=[s0, s1, s2])
c = gf.path.extrude(P, X)
c.plot()
```


## Custom curves

Now let us have some fun and try to make a loop-de-loop structure with parallel
straights and several ports.

To create a new type of curve we simply make a function that produces an array
of points. The best way to do that is to create a function which allows you to
specify a large number of points along that curve -- in the case shown below, the
`looploop()` function outputs 1000 points along a looping path.  Later, if we
want to reduce the number of points in our geometry we can easily `simplify` the
path.


```python
def looploop(num_pts=1000):
    """Simple limacon looping curve."""

    # This line creates an array of num_pts evenly spaced numbers ranging from -π to 0. This array represents the angle t in polar coordinates.
    t = np.linspace(-np.pi, 0, num_pts) 
    r = 20 + 25 * np.sin(t) # This line calculates the radius r for each corresponding angle t using the polar equation for a limaçon curve.

    # # These lines convert the polar coordinates (r, t) into standard Cartesian coordinates (x, y), which are needed for plotting.
    x = r * np.cos(t) 
    y = r * np.sin(t) 

    # The separate x and y arrays are combined into a single NumPy array of coordinate pairs, which is then returned by the function.
    return np.array((x, y)).T


# Create the path points.
P = gf.Path()
P.append(gf.path.arc(radius=10, angle=90))
P.append(gf.path.straight())
P.append(gf.path.arc(radius=5, angle=-90))
P.append(looploop(num_pts=1000))
P.rotate(-45)

# Create the cross-section.
s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
s1 = gf.Section(width=0.5, offset=2, layer=(2, 0))
s2 = gf.Section(width=0.5, offset=4, layer=(3, 0))
s3 = gf.Section(width=1, offset=0, layer=(4, 0))
X = gf.CrossSection(sections=(s0, s1, s2, s3))

c = gf.path.extrude(P, X)
c.plot()
```

You can create Paths from any array of points -- just be sure that they form
smooth curves!  If we examine our path `P` we can see that we have effortlessly
created a long list of points:

```python
path_points = P.points  # Curve points are stored as a numpy array in P.points.
print(np.shape(path_points))  # The shape of the array is Nx2.
print(len(P))  # Equivalently, use len(P) to see how many points are inside.
```

## Simplifying / reducing point usage

One of the primary concerns of generating smooth curves is that too many points
are generated, inflating file sizes and making boolean operations
computationally expensive. Fortunately, PHIDL has a fast implementation of the
[Ramer-Douglas–Peucker
algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
that lets you reduce the number of points in a curve without changing its shape.
All that needs to be done when you make a `component()` is extruding the path with a cross_section, you need to specify the
`simplify` argument.

If we specify `simplify = 1e-3`, the number of points in the line drops from
12,000 to 4,000, and the remaining points form a line that is identical to
within `1e-3` distance from the original (for the default 1 micron unit size,
this corresponds to 1 nanometer resolution):

```python
# The remaining points form a identical line to within `1e-3` from the original.
c = gf.path.extrude(p=P, cross_section=X, simplify=1e-3)
c.plot()
```

Let us say we need fewer points.  We can increase the `simplify` tolerance by specifying `simplify = 1e-1`.  This drops the number of points to ~400 points and they form a line that is identical to within `1e-1` distance from the original:

```python
c = gf.path.extrude(P, cross_section=X, simplify=1e-1)
c.plot()
```

Taken to absurdity, what happens if we set `simplify = 0.3`?  Once again, the
~200 remaining points form a line that is within `0.3` units from the original
-- but that line will look pretty bad.

```python
c = gf.path.extrude(P, cross_section=X, simplify=0.3)
c.plot()
```

## Curvature calculation

The `Path` class has a `curvature()` method that computes the curvature `K` of
your smooth path (K = 1/(radius of curvature)).  This can be helpful for
verifying that your curves transition smoothly such as in [track-transition
curves](https://en.wikipedia.org/wiki/Track_transition_curve) (also known as
"Euler" bends in the photonics world). Euler bends have lower mode-mismatch loss as explained in [this paper](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-27-22-31394&id=422321)

Note this curvature is numerically computed, so areas in which the curvature jumps
instantaneously (such as between an arc and a straight segment) will be slightly
interpolated, and sudden changes in point density along the curve can cause
discontinuities.

```python
straight_points = 100

P = gf.Path()
P.append(
    [
        gf.path.straight(
            length=10, npoints=straight_points
        ),  # Should have a curvature of 0
        gf.path.euler(
            radius=3, angle=90, p=0.5, use_eff=False
        ),  # Euler straight-to-bend transition with min. bend radius of 3 (max curvature of 1/3)
        gf.path.straight(
            length=10, npoints=straight_points
        ),  # Should have a curvature of 0
        gf.path.arc(radius=10, angle=90),  # Should have a curvature of 1/10
        gf.path.arc(radius=5, angle=-90),  # Should have a curvature of -1/5
        gf.path.straight(
            length=2, npoints=straight_points
        ),  # Should have a curvature of 0
    ]
)

f = P.plot()
```

Arc paths are equivalent to `bend_circular` and euler paths are equivalent to `bend_euler`.

```python
# The .curvature() method of the Path object P is called.
# It returns two arrays: s, which contains the cumulative distance (arc length) at each point along the path,
# and K, which contains the corresponding curvature at that point. (Curvature is the reciprocal of the bend radius).
s, K = P.curvature()

# This plots the arc length s on the x-axis and the curvature K on the y-axis.
# The ".-" format string specifies that the plot should be a line with a dot marker at each data point.
plt.plot(s, K, ".-")
plt.xlabel("Position along curve (arc length)")
plt.ylabel("Curvature")
```

```python
P = gf.path.euler(radius=3, angle=90, p=1.0, use_eff=False)
P.append(gf.path.euler(radius=3, angle=90, p=0.2, use_eff=False))
P.append(gf.path.euler(radius=3, angle=90, p=0.0, use_eff=False))
P.plot()
```

```python
s, K = P.curvature()
plt.plot(s, K, ".-")
plt.xlabel("Position along curve (arc length)")
plt.ylabel("Curvature")
```

You can compare two 90 degrees euler bends with 180 euler bend.

A 180 euler bend is shorter, and has less loss than two 90 degrees euler bend.

```python
straight_points = 100

P = gf.Path()
P.append(
    [
        gf.path.euler(radius=3, angle=90, p=1, use_eff=False),
        gf.path.euler(radius=3, angle=90, p=1, use_eff=False),
        gf.path.straight(length=6, npoints=100),
        gf.path.euler(radius=3, angle=180, p=1, use_eff=False),
    ]
)

f = P.plot()
```

```python
s, K = P.curvature()
plt.plot(s, K, ".-")
plt.xlabel("Position along curve (arc length)")
plt.ylabel("Curvature")
```

## Transitioning between cross-sections

Often a critical element of building paths is being able to transition between
cross-sections.  You can use the `transition()` function to do exactly this: You
simply feed it two `CrossSection`s and it will output a new `CrossSection` that
smoothly transitions between the two.

Let us start off by creating two cross-sections we want to transition between.
Note we give all the cross-sectional elements names by specifying the `name`
argument in the `add()` function -- this is important because the transition
function will try to match names between the two input cross-sections, and any
names not present in both inputs will be skipped.

```python
# Create our first Cross-section.
import gdsfactory as gf

s0 = gf.Section(width=1.2, offset=0, layer=(2, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=2.2, offset=0, layer=(3, 0), name="etch")
s2 = gf.Section(width=1.1, offset=3, layer=(1, 0), name="wg2")
X1 = gf.CrossSection(sections=[s0, s1, s2])

# Create the second Cross-section that we want to transition to.
s0 = gf.Section(width=1, offset=0, layer=(2, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=3.5, offset=0, layer=(3, 0), name="etch")
s2 = gf.Section(width=3, offset=5, layer=(1, 0), name="wg2")
X2 = gf.CrossSection(sections=[s0, s1, s2])

# To show the cross-sections, let us now create two paths and create components by extruding them.
P1 = gf.path.straight(length=5)
P2 = gf.path.straight(length=5)
wg1 = gf.path.extrude(P1, X1)
wg2 = gf.path.extrude(P2, X2)

# Place both cross-section components and quickplot them,
# Quickplot is designed to create a wide variety of complex graphs with a simple, concise syntax, making it ideal for quick data exploration.
c = gf.Component()
wg1ref = c << wg1
wg2ref = c << wg2
wg2ref.movex(7.5)

c.plot()
```

Now we can create the transitional cross-section by calling a `transition()` with
these two cross-sections as the input. If we want the width to vary as a smooth
sinusoid between the sections, we can set `width_type` to `'sine'`
(alternatively we could also use `'linear'`).

```python
# Create the transitional cross-section.
Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")

# Create a Path for the transitional cross-section to follow.
P3 = gf.path.straight(length=15, npoints=100)

# Use the transitional cross-section to create a component.
straight_transition = gf.path.extrude_transition(P3, Xtrans)
straight_transition.plot()
```

Now that we have all of our components, let us proceed to `connect()` everything and see
what it looks like:

```python
c = gf.Component("transition_demo")

wg1ref = c << wg1
wgtref = c << straight_transition
wg2ref = c << wg2

wgtref.connect("o1", wg1ref.ports["o2"])
wg2ref.connect("o1", wgtref.ports["o2"])

c.plot()
```

Note that since `transition()` outputs a `Transition`, we can make the transition follow an arbitrary path:

```python
# Transition along a curving path.
P4 = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
wg_trans = gf.path.extrude_transition(P4, Xtrans)

c = gf.Component("demo_transition")
wg1_ref = c << wg1  # First cross-section component.
wg2_ref = c << wg2
wgt_ref = c << wg_trans

wgt_ref.connect("o1", wg1_ref.ports["o2"])
wg2_ref.connect("o1", wgt_ref.ports["o2"])

c.plot()
```


You can also extrude an arbitrary transition:

```python
w1 = 1
w2 = 5
x1 = gf.get_cross_section("strip", width=w1)
x2 = gf.get_cross_section("strip", width=w2)
transition = gf.path.transition(x1, x2)
p = gf.path.arc(radius=10)
c = gf.path.extrude_transition(p, transition)
c.plot()
```

### Asymmetric transition

In some cases, you may want the edges of the transition to follow a different function.
This can be done by using the `transition_asymmetric()` function.
In this case, the argument `width_type` of `transition` is split into `width_type1`, corresponding to the lower edge, and `width_type2`, corresponding to the upper edge of the transition.
As in the case of `transition()`, the user can define their own transition function.

Let us look at an example where the upper edge follows the sinusoidal (default) transition of the width, while the lower follows a user-defined polynomial.

```python
import gdsfactory as gf

# Define a custom polynomial transition function from y1 -> y2, for t ∈ [0,1].
def polynomial(t: float, y1: float, y2: float) -> float:
        return (y2 - y1) * t**3 + y1

w1 = 2
w2 = 6
length = 10
cs1 = gf.get_cross_section("strip", width=w1)
cs2 = gf.get_cross_section("strip", width=w2)

transition = gf.path.transition_asymmetric(
    cs1, cs2, width_type1=polynomial, width_type2="sine")
p = gf.path.straight(length, npoints=100)
c = gf.path.extrude_transition(p, transition)

c.plot()
```

## Variable width / offset

In some instances, you may want to vary the width or offset of the path's cross-section as it travels.
This can be accomplished by giving the `CrossSection`
arguments that are functions or lists.  Let us say we wanted a width that varies
sinusoidally along the length of the Path.  To do this, we need to make a width
function that is parameterized from 0 to 1: for an example function
`my_width_fun(t)` where the width at `t==0` is the width at the beginning of the
path and the width at `t==1` is the width at the end.


```python
import numpy as np

import gdsfactory as gf


def my_custom_width_fun(t):
    # Note: Custom width/offset functions MUST be vectorizable --
    # you must be able to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4]).
    num_periods = 5

    # np.cos(...): This is the core cosine function from the NumPy library, which generates a wave that oscillates between -1 and 1.
    # 2 * np.pi * t * num_periods: This part calculates the angle (in radians) for the cosine function.
    # It determines the frequency of the wave, i.e., how many full cycles (num_periods) it completes over a given time t.
    # This adds a vertical offset of 3 to the wave. Instead of oscillating between -1 and 1, the wave now oscillates between 2 (3 - 1) and 4 (3 + 1).
    return 3 + np.cos(2 * np.pi * t * num_periods)


P = gf.path.straight(length=40, npoints=30)

#Create two cross-sections: one fixed width, one modulated by my_custom_offset_fun.
s0 = gf.Section(width=3, offset=-6, layer=(2, 0))
s1 = gf.Section(width=0, width_function=my_custom_width_fun, offset=0, layer=(1, 0))
X = gf.CrossSection(sections=(s0, s1))

# # Extrude the path to create the component.
c = gf.path.extrude(P, cross_section=X)
c.plot()
```


We can do the same thing with the offset argument:


```python
def my_custom_offset_fun(t):
    num_periods = 3
    return 3 + np.cos(2 * np.pi * t * num_periods)


P = gf.path.straight(length=40, npoints=30)

s0 = gf.Section(width=1, offset=0, layer=(1, 0))
s1 = gf.Section(
    width=1,
    offset_function=my_custom_offset_fun,
    layer=(2, 0),
    port_names=("clad1", "clad2"),
)
X = gf.CrossSection(sections=(s0, s1))

c = gf.path.extrude(P, cross_section=X)
c.plot()
```


## Offsetting a Path

Sometimes it is convenient to start with a simple path and offset the line it
follows to suit your needs (without using a custom-offset cross-section). Here,
we start with two copies of a simple straight path and use the `offset()`
function to directly modify each path.


```python
def my_custom_offset_fun(t):

    num_periods = 3
    return 2 + np.cos(2 * np.pi * t * num_periods)


P1 = gf.path.straight(npoints=101)
P1.offset(offset=my_custom_offset_fun)
f = P1.plot()
```

```python
P2 = P1.copy()  # Make a copy of the path.
P2.mirror((1, 0))  # Mirror across X-axis.
f2 = P2.plot()
```

```python
P = gf.path.arc(radius=10, angle=45)

s0 = gf.Section(width=1, offset=3, layer=(2, 0), name="waveguide")
s1 = gf.Section(width=1, offset=0, layer=(1, 0), name="heater", port_names=("o1", "o2"))
X = gf.CrossSection(sections=(s0, s1))
c = gf.path.extrude(P, X)
c.plot()
```

```python
P = gf.Path()
P.append(gf.path.arc(radius=10, angle=90))  # Circular arc.
P.append(gf.path.straight(length=10))  # Straight section.
P.append(gf.path.euler(radius=3, angle=-90))  # Euler bend (aka "racetrack" curve).
P.append(gf.path.straight(length=40))
P.append(gf.path.arc(radius=8, angle=-45))
P.append(gf.path.straight(length=10))
P.append(gf.path.arc(radius=8, angle=45))
P.append(gf.path.straight(length=10))

f = P.plot()
```

```python
c = gf.path.extrude(P, width=1, layer=(2, 0))
c.plot()
```

```python
s0 = gf.Section(width=2, offset=0, layer=(2, 0))
xs = gf.CrossSection(sections=(s0,))
c = gf.path.extrude(P, xs)
c.plot()
```

```python
p = gf.path.straight(length=10, npoints=101)
s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("o1", "o2"), name="core")
s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
x1 = gf.CrossSection(sections=(s0, s1))
c = gf.path.extrude(p, x1)
c.plot()
```

```python
s0 = gf.Section(
    width=1 + 3, offset=0, layer=(1, 0), port_names=("o1", "o2"), name="core"
)
s1 = gf.Section(width=3 + 3, offset=0, layer=(3, 0), name="slab")
x2 = gf.CrossSection(sections=(s0, s1))
c2 = gf.path.extrude(p, x2)
c2.plot()
```

```python
t = gf.path.transition(x1, x2)
c3 = gf.path.extrude_transition(p, t)
c3.plot()
```

```python
c4 = gf.Component()
start_ref = c4 << c
trans_ref = c4 << c3
end_ref = c4 << c2

trans_ref.connect("o1", start_ref.ports["o2"])
end_ref.connect("o1", trans_ref.ports["o2"])
c4.plot()
```

### Avoiding transitions for specific layers

`transition()` and `extrude_transition()` match sections between the two cross-sections **by `name`**. Only sections whose `name` appears in **both** cross-sections will be transitioned — any section present in only one cross-section is skipped.

You can use this to avoid transitioning a specific layer: simply give it a different `name` in each cross-section (or omit it from one).

```python
import gdsfactory as gf

p = gf.path.straight(length=10, npoints=101)

# Cross-section 1: core + slab (both named)
s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
x1 = gf.CrossSection(sections=(s0, s1))

# Cross-section 2: wider core + wider slab
s0 = gf.Section(width=1.0, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=5, offset=0, layer=(3, 0), name="slab")
x2 = gf.CrossSection(sections=(s0, s1))

# Both "core" and "slab" are transitioned
t_both = gf.path.transition(x1, x2, width_type="linear")
c_both = gf.path.extrude_transition(p, t_both)
c_both.plot()
```

```python
# Now avoid transitioning the slab by giving it different names
s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab_in")  # different name
x1_no_slab = gf.CrossSection(sections=(s0, s1))

s0 = gf.Section(width=1.0, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=5, offset=0, layer=(3, 0), name="slab_out")  # different name
x2_no_slab = gf.CrossSection(sections=(s0, s1))

# Only "core" is transitioned, slab sections are skipped
t_core_only = gf.path.transition(x1_no_slab, x2_no_slab, width_type="linear")
c_core_only = gf.path.extrude_transition(p, t_core_only)
c_core_only.plot()
```

```python
# Or use skip_transition=True on the Section you want to keep constant
s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab", skip_transition=True)
x1_skip = gf.CrossSection(sections=(s0, s1))

s0 = gf.Section(width=1.0, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2"))
s1 = gf.Section(width=5, offset=0, layer=(3, 0), name="slab", skip_transition=True)
x2_skip = gf.CrossSection(sections=(s0, s1))

# Only "core" is transitioned, slab is skipped
t_skip = gf.path.transition(x1_skip, x2_skip, width_type="linear")
c_skip = gf.path.extrude_transition(p, t_skip)
c_skip.plot()
```

## Creating new cross_sections

You can create functions that return a cross_section in 2 ways:

- Customize an existing cross-section for example `gf.cross_section.strip`.
- Define a function that returns a cross_section.
- Define a CrossSection object.

What parameters do `cross_section` take?

```python
help(gf.cross_section.cross_section)
```

```python
import gdsfactory as gf
from gdsfactory.cross_section import CrossSection, cross_section, xsection
from gdsfactory.typings import LayerSpec

@xsection
def pin(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    radius: float = 10.0,
    radius_min: float = 5,
    layer_p: LayerSpec = (21, 0),
    layer_n: LayerSpec = (20, 0),
    width_p: float = 2,
    width_n: float = 2,
    offset_p: float = 1,
    offset_n: float = -1,
    **kwargs,
) -> CrossSection:
    """Return PIN cross_section."""
    sections = (
        gf.Section(layer=layer_p, width=width_p, offset=offset_p),
        gf.Section(layer=layer_n, width=width_n, offset=offset_n),
    )

    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )
```

```python
c = gf.components.straight(cross_section=pin)
c.plot()
```

```python
pin5 = gf.components.straight(cross_section=pin, length=5)
pin5.plot()
```

```python
pin5 = gf.components.straight(cross_section="pin", length=5)
pin5.plot()
```

Finally, you can also pass the dictionary (dict) of most components that define the cross-section.

```python
# Create our first cross-section
s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), name="wg", port_names=("o1", "o2"))
s1 = gf.Section(width=0.2, offset=0, layer=(3, 0), name="slab")
x1 = gf.CrossSection(sections=(s0, s1))

# Create the second cross-section that we want to transition to.
s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), name="wg", port_names=("o1", "o2"))
s1 = gf.Section(width=3.0, offset=0, layer=(3, 0), name="slab")
x2 = gf.CrossSection(sections=(s0, s1))

# To show the cross-sections, let us create two paths and create components by extruding them.
p1 = gf.path.straight(length=5)
p2 = gf.path.straight(length=5)
wg1 = gf.path.extrude(p1, x1)
wg2 = gf.path.extrude(p2, x2)

# Place both cross-section components and quickplot them.
c = gf.Component()
wg1ref = c << wg1
wg2ref = c << wg2
wg2ref.movex(7.5)

# Create the transitional cross-section.
xtrans = gf.path.transition(cross_section1=x1, cross_section2=x2, width_type="linear")
# Create a path for the transitional cross-section to follow.
p3 = gf.path.straight(length=15, npoints=100)

# Use the transitional cross-section to create a component.
straight_transition = gf.path.extrude_transition(p3, xtrans)
straight_transition.plot()
```

```python

xtrans = gf.path.transition(
    cross_section1=x1, cross_section2=x2, width_type="parabolic"
)

p3 = gf.path.straight(length=15, npoints=100)


straight_transition = gf.path.extrude_transition(p3, xtrans)
straight_transition.plot()
```

```python

xtrans = gf.path.transition(cross_section1=x1, cross_section2=x2, width_type="sine")
p3 = gf.path.straight(length=15, npoints=100)


straight_transition = gf.path.extrude_transition(p3, xtrans)
straight_transition.plot()
```

```python
s = straight_transition.to_3d()
s.show()
```

The port location, width and orientation remains the same for a sheared component. However, an additional property, `shear_angle` is set to the value of the shear angle. In general, shear ports can be safely connected together.


## bbox_layers vs cladding_layers

For extruding waveguides you have two options:

1. bbox_layers for squared bounding box.
2. cladding_layers for extruding a layer that follows the shape of the path.

```python
xs_bbox = gf.cross_section.cross_section(bbox_layers=((3, 0),), bbox_offsets=(3,))
w1 = gf.components.bend_euler(cross_section=xs_bbox)
w1.plot()
```

```python
xs_clad = gf.cross_section.cross_section(cladding_layers=[(3, 0)], cladding_offsets=[3])
w2 = gf.components.bend_euler(cross_section=xs_clad)
w2.plot()
```

## Insets

It is handy to be able to extrude a `CrossSection` along a `Path`, while each `Section` may have a particular inset relative to the main `Section`. An example of this is a waveguide with a heater.

```python
import gdsfactory as gf


@xsection
def xs_waveguide_heater() -> gf.CrossSection:
    return gf.cross_section.cross_section(
        layer="WG",
        width=0.5,
        sections=(
            gf.cross_section.Section(
                name="heater",
                width=1,
                layer="HEATER",
                insets=(1, 2),
            ),
        ),
    )


c = gf.components.straight(cross_section=xs_waveguide_heater)
c.plot()
```

```python
@xsection
def xs_waveguide_heater_with_ports() -> gf.CrossSection:
    return gf.cross_section.cross_section(
        layer="WG",
        width=0.5,
        sections=(
            gf.cross_section.Section(
                name="heater",
                width=1,
                layer="HEATER",
                insets=(1, 2),
                port_names=("e1", "e2"),
                port_types=("electrical", "electrical"),
            ),
        ),
    )


c = gf.components.straight(cross_section=xs_waveguide_heater_with_ports)
c.plot()
```
