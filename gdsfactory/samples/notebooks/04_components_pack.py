# # Grid / pack / align / distribute

# ## Grid
#
#
# The ``gf.components.grid()`` function can take a list (or 2D array) of objects and arrange them along a grid. This is often useful for making parameter sweeps.   If the `separation` argument is true, grid is arranged such that the elements are guaranteed not to touch, with a `spacing` distance between them.  If `separation` is false, elements are spaced evenly along a grid. The `align_x`/`align_y` arguments specify intra-row/intra-column alignment.  The `edge_x`/`edge_y` arguments specify inter-row/inter-column alignment (unused if `separation = True`).

# +
import gdsfactory as gf

components_list = []
for width1 in [1, 6, 9]:
    for width2 in [1, 2, 4, 8]:
        D = gf.components.taper(length=10, width1=width1, width2=width2, layer=(1, 0))
        components_list.append(D)

c = gf.grid(
    components_list,
    spacing=(5, 1),
    separation=True,
    shape=(3, 4),
    align_x="x",
    align_y="y",
    edge_x="x",
    edge_y="ymax",
)
c.plot()
# -

# ## Pack
#
#
# The ``gf.pack()`` function packs geometries together into rectangular bins. If a ``max_size`` is specified, the function will create as many bins as is necessary to pack all the geometries and then return a list of the filled-bin Components.
#
# Here we generate several random shapes then pack them together automatically. We allow the bin to be as large as needed to fit all the Components by specifying ``max_size = (None, None)``.  By setting ``aspect_ratio = (2,1)``, we specify the rectangular bin it tries to pack them into should be twice as wide as it is tall:

# +
import numpy as np
import gdsfactory as gf

np.random.seed(5)
D_list = [gf.components.rectangle(size=(i, i)) for i in range(1, 10)]

D_packed_list = gf.pack(
    D_list,  # Must be a list or tuple of Components
    spacing=1.25,  # Minimum distance between adjacent shapes
    aspect_ratio=(2, 1),  # (width, height) ratio of the rectangular bin
    max_size=(None, None),  # Limits the size into which the shapes will be packed
    density=1.05,  # Values closer to 1 pack tighter but require more computation
    sort_by_area=True,  # Pre-sorts the shapes by area
)
D = D_packed_list[0]  # Only one bin was created, so we plot that
D
# -

# Say we need to pack many shapes into multiple 500x500 unit die. If we set ``max_size = (500,500)`` the shapes will be packed into as many 500x500 unit die as required to fit them all:

# +
np.random.seed(1)
D_list = [
    gf.components.ellipse(radii=tuple(np.random.rand(2) * n + 2)) for n in range(120)
]
D_packed_list = gf.pack(
    D_list,  # Must be a list or tuple of Components
    spacing=4,  # Minimum distance between adjacent shapes
    aspect_ratio=(1, 1),  # Shape of the box
    max_size=(500, 500),  # Limits the size into which the shapes will be packed
    density=1.05,  # Values closer to 1 pack tighter but require more computation
    sort_by_area=True,  # Pre-sorts the shapes by area
)

# Put all packed bins into a single device and spread them out with distribute()
F = gf.Component("packed")
[F.add_ref(D) for D in D_packed_list]
F.distribute(elements="all", direction="x", spacing=100, separation=True)
F
# -

# Note that the packing problem is an NP-complete problem, so ``gf.components.packer()`` may be slow if there are more than a few hundred Components to pack (in that case, try pre-packing a few dozen at a time then packing the resulting bins). Requires the ``rectpack`` python package.

# ## Distribute
#
#
# The ``distribute()`` function allows you to space out elements within a Component evenly in the x or y direction.  It is meant to duplicate the distribute functionality present in Inkscape / Adobe Illustrator:


# ![](https://i.imgur.com/dC74M8x.png)

# Say we start out with a few random-sized rectangles we want to space out:

c = gf.Component("rectangles")
# Create different-sized rectangles and add them to D
[
    c.add_ref(
        gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20], layer=(2, 0))
    ).move([n, n * 4])
    for n in [0, 2, 3, 1, 2]
]
c.plot()

# Oftentimes, we want to guarantee some distance between the objects.  By setting ``separation = True`` we move each object such that there is ``spacing`` distance between them:

D = gf.Component("rectangles_separated")
# Create different-sized rectangles and add them to D
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move((n, n * 4))
    for n in [0, 2, 3, 1, 2]
]
# Distribute all the rectangles in D along the x-direction with a separation of 5
D.distribute(
    elements="all",  # either 'all' or a list of objects
    direction="x",  # 'x' or 'y'
    spacing=5,
    separation=True,
)
D

# Alternatively, we can spread them out on a fixed grid by setting ``separation = False``. Here we align the left edge (``edge = 'min'``) of each object along a grid spacing of 100:

D = gf.Component("spacing100")
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move((n, n * 4))
    for n in [0, 2, 3, 1, 2]
]
D.distribute(
    elements="all", direction="x", spacing=100, separation=False, edge="xmin"
)  # edge must be either 'xmin' (left), 'xmax' (right), or 'x' (center)
D

# The alignment can be done along the right edge as well by setting ``edge = 'max'``, or along the center by setting ``edge = 'center'`` like in the following:

D = gf.Component("alignment")
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move(
        (n - 10, n * 4)
    )
    for n in [0, 2, 3, 1, 2]
]
D.distribute(
    elements="all", direction="x", spacing=100, separation=False, edge="x"
)  # edge must be either 'xmin' (left), 'xmax' (right), or 'x' (center)
D

# ## Align
#
#
# The ``align()`` function allows you to elements within a Component horizontally or vertically.  It is meant to duplicate the alignment functionality present in Inkscape / Adobe Illustrator:

# ![](https://i.imgur.com/rqzunXM.png)

# Say we ``distribute()`` a few objects, but they're all misaligned:

D = gf.Component("distribute")
# Create different-sized rectangles and add them to D then distribute them
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move((n, n * 4))
    for n in [0, 2, 3, 1, 2]
]
D.distribute(elements="all", direction="x", spacing=5, separation=True)
D

# we can use the ``align()`` function to align their top edges (``alignment = 'ymax'):

# +
D = gf.Component("align")
# Create different-sized rectangles and add them to D then distribute them
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move((n, n * 4))
    for n in [0, 2, 3, 1, 2]
]
D.distribute(elements="all", direction="x", spacing=5, separation=True)

# Align top edges
D.align(elements="all", alignment="ymax")
D
# -

# or align their centers (``alignment = 'y'):

# +
D = gf.Component("distribute_align_y")
# Create different-sized rectangles and add them to D then distribute them
[
    D.add_ref(gf.components.rectangle(size=[n * 15 + 20, n * 15 + 20])).move((n, n * 4))
    for n in [0, 2, 3, 1, 2]
]
D.distribute(elements="all", direction="x", spacing=5, separation=True)

# Align top edges
D.align(elements="all", alignment="y")
D
# -

# other valid alignment options include ``'xmin', 'x', 'xmax', 'ymin', 'y', and 'ymax'``
