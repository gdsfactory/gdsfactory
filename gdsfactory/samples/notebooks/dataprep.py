# # Maskprep
#
# When building a reticle sometimes you want to do boolean operations. This is usually known as maskprep or dataprep.
#
# You can do this at the component level or at the top reticle assembled level, each having different advantages and disadvantages.
#
#
# ## Component level
#
#
# Lets try to remove acute angles that can cause min space DRC violations (Design Rule Checking). This happens a lot when you have cladding layers and couplers.
#
# ### Remove acute angles

# +
from gdsfactory.generic_tech.layer_map import LAYER as l
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.generic_tech.get_generic_pdk()
PDK.activate()

# +
from functools import partial
import gdsfactory as gf
from gdsfactory.geometry.maskprep import get_polygons_over_under, over_under

c = gf.components.coupler_ring(
    cladding_layers=((2, 0),),
    cladding_offsets=(0.2,),
)
c.plot()
# -


# ### Decorator
#
# A decorator is a function that calls a function [see Python intro](https://gdsfactory.github.io/gdsfactory/notebooks/_0_python.html) or read some python books.

# +
over_under_slab = partial(over_under, layers=((2, 0),), distances=(0.5,))

c = gf.components.coupler_ring(
    cladding_layers=((2, 0)),
    cladding_offsets=(0.2,),
    decorator=over_under_slab,
)
c.plot()
# -

# ### Get polygons
#
# You can also remove acute angles by adding extra polygons on top.

# +
get_polygons_over_under_slab = partial(
    get_polygons_over_under, layers=((2, 0)), distances=(0.5,)
)

c = gf.Component("compnent_clean")
ref = c << gf.components.coupler_ring(
    cladding_layers=((2, 0)),
    cladding_offsets=(0.2,),  # decorator=over_under_slab_decorator
)
polygons = get_polygons_over_under_slab(ref)
c.add(polygons)
c.plot()
# -


# ### Invert tone
#
# Sometimes you need to define not what you keep (positive resist) but what you etch (negative resist).
# We have some useful functions to invert the tone.

c = gf.components.add_trenches(component=gf.components.coupler)
c.plot()

c = gf.components.add_trenches(component=gf.components.ring_single)
c.plot()

c = gf.components.add_trenches(
    component=gf.components.grating_coupler_elliptical_lumerical(layer_slab=None)
)
c.plot()


c = gf.components.add_trenches90(component=gf.components.bend_euler(radius=20))
c.plot()

# ## Flatten top level
#
# You can flatten the hierarchy and use klayout LayerProcessor to create a `RegionCollection` where you can easily grow and shrink layers.
#
# The advantage is that this can easily clean up routing, proximity effects, boolean operations on big masks.
#
# The disadvantage is that the design is no longer hierarchical and can take up more space.
#
# ### Size
#
# You can copy/size layers

# +
c = gf.Component()

device = c << gf.components.coupler_ring()
floorplan = c << gf.components.bbox(device.bbox, layer=l.FLOORPLAN)
c.write_gds("src.gds")
c.plot()

# +
import gdsfactory.geometry.maskprep_flat as dp

d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()  # copy layer
d.SLAB150 += 4  # size layer by 4 um
d.SLAB150 -= 2  # size layer by 2 um
c = d.write("dst.gds")
c.plot()
# -

# ### Booleans
#
# You can derive layers and do boolean operations.
#

d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150 += 3  # size layer by 3 um
d.SHALLOW_ETCH = d.SLAB150 - d.WG
c = d.write("dst.gds")
c.plot()


# ### Fill
#
# You can add rectangular fill, using booleans to decide where to add it:

# +
d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))

fill_region = d.FLOORPLAN - d.WG
fill_cell = d.get_fill(
    fill_region,
    size=[0.1, 0.1],
    spacing=[0.1, 0.1],
    fill_layers=[l.WG, l.M1],
    fill_name="test",
)
fill_cell
# -

# ### KLayout operations
#
# Any operation from Klayout Region can be called directly:

d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
d.SLAB150 = d.WG.copy()
d.SLAB150.round_corners(1 * 1e3, 1 * 1e3, 100)  # round corners by 1um
c = d.write("dst.gds")
c.plot()
