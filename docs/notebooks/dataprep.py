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
import gdsfactory as gf
from gdsfactory.generic_tech.layer_map import LAYER as l
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
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
