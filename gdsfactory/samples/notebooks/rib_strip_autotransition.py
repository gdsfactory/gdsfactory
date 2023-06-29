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
# # Routing with different CrossSections
#
# When working in a technologies with multiple waveguide cross-sections, it is useful to differentiate intent layers for the different waveguide types
# and assign default transitions between those layers. In this way, you can easily autotransition between the different cross-section types.
#
# ## Setting up your PDK
#
# Let's first set up a sample PDK with the following key features:
#
# 1. Rib and strip cross-sections with differentiated intent layers.
# 2. Default transitions for each individual cross-section type (width tapers), and also a rib-to-strip transition component to switch between them.
# 3. Preferred routing cross-sections defined for the all-angle router.

# %%
from gdsfactory.decorators import has_valid_transformations
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from functools import partial
from gdsfactory.cross_section import strip, rib_conformal
from gdsfactory.typings import CrossSectionSpec
from gdsfactory.routing import all_angle
from gdsfactory.read import cell_from_yaml_template

gf.clear_cache()
gf.config.rich_output()
generic_pdk = get_generic_pdk()
generic_pdk.circuit_yaml_parser = cell_from_yaml_template

# define our rib and strip waveguide intent layers
RIB_INTENT_LAYER = (2000, 11)
STRIP_INTENT_LAYER = (2001, 11)

# create strip and rib cross-sections, with differentiated intent layers
strip_with_intent = partial(
    strip,
    layer="STRIP_INTENT",
    cladding_layers=["WG"],  # keeping WG layer is nice for compatibility
    cladding_offsets=[0],
    gap=2,
)

rib_with_intent = partial(
    rib_conformal,
    layer="RIB_INTENT",
    cladding_layers=["WG"],  # keeping WG layer is nice for compatibility
    cladding_offsets=[0],
    gap=5,
)


# create strip->rib transition component
@gf.cell
def strip_to_rib(width1: float = 0.5, width2: float = 0.5):
    c = gf.Component()
    taper = c << gf.c.taper_strip_to_ridge(width1=width1, width2=width2)
    c.add_port(
        "o1",
        port=taper.ports["o1"],
        layer="STRIP_INTENT",
        cross_section=strip_with_intent(width=width1),
    )
    c.add_port(
        "o2",
        port=taper.ports["o2"],
        layer="RIB_INTENT",
        cross_section=rib_with_intent(width=width2),
    )
    c.info.update(taper.info)
    return c


# also define a rib->strip component for transitioning the other way
@gf.cell
def rib_to_strip(width1: float = 0.5, width2: float = 0.5):
    c = gf.Component()
    taper = c << strip_to_rib(width1=width2, width2=width1)
    c.add_port("o1", port=taper.ports["o2"])
    c.add_port("o2", port=taper.ports["o1"])
    c.info.update(taper.info)
    return c


# create single-layer taper components
@gf.cell
def taper_single_cross_section(
    cross_section: CrossSectionSpec = "strip", width1: float = 0.5, width2: float = 1.0
):
    cs1 = gf.get_cross_section(cross_section, width=width1)
    cs2 = gf.get_cross_section(cross_section, width=width2)
    length = abs(width1 - width2) * 10
    c = gf.c.taper_cross_section_linear(cs1, cs2, length=length).copy()
    c.info["length"] = length
    return c


taper_strip = partial(taper_single_cross_section, cross_section="strip")
taper_rib = partial(taper_single_cross_section, cross_section="rib")

# make a new PDK with our required layers, cross-sections, and default transitions
multi_wg_pdk = gf.Pdk(
    base_pdk=generic_pdk,
    name="multi_wg_demo",
    layers={
        "RIB_INTENT": RIB_INTENT_LAYER,
        "STRIP_INTENT": STRIP_INTENT_LAYER,
    },
    cross_sections={
        "rib": rib_with_intent,
        "strip": strip_with_intent,
    },
    layer_transitions={
        RIB_INTENT_LAYER: taper_rib,
        STRIP_INTENT_LAYER: taper_strip,
        (RIB_INTENT_LAYER, STRIP_INTENT_LAYER): rib_to_strip,
        (STRIP_INTENT_LAYER, RIB_INTENT_LAYER): strip_to_rib,
    },
    layer_views=generic_pdk.layer_views,
)

# activate our new PDK
multi_wg_pdk.activate()

# set to prefer rib routing when there is enough space
all_angle.LOW_LOSS_CROSS_SECTIONS.insert(0, "rib")

# %% [markdown]
# Let's quickly demonstrate our new cross-sections and transition component.

# %%
# demonstrate rib and strip waveguides in our new PDK
strip_width = 1
rib_width = 0.7
c = gf.Component()
strip_wg = c << gf.c.straight(cross_section="strip", width=strip_width)
rib_wg = c << gf.c.straight(cross_section="rib", width=rib_width)
taper = c << strip_to_rib(width1=strip_width, width2=rib_width)
taper.connect("o1", strip_wg.ports["o2"])
rib_wg.connect("o1", taper.ports["o2"])
c.plot()

# %% [markdown]
# ## Autotransitioning with the All-Angle Router
#
# Now that our PDK and settings are all configured, we can see how the all-angle router will
# auto-transition for us between different cross sections.
#
# Because we are using the low-loss connector by default, and the highest priority cross section is rib,
# we will see rib routing anywhere there is enough space to transition.

# %%
from gdsfactory.read import cell_from_yaml_template
from IPython.display import Code
from pathlib import Path
from IPython.display import display


def show_yaml_pic(filepath):
    gf.clear_cache()
    cell_name = filepath.stem
    return display(
        Code(filename=filepath, language="yaml+jinja"),
        cell_from_yaml_template(filepath, name=cell_name)(),
    )


# load a yaml PIC, and see how it looks with our new technology
sample_dir = Path("yaml_pics")

basic_sample_fn = sample_dir / "aar_indirect.pic.yml"
show_yaml_pic(basic_sample_fn)


# %%
c = gf.read.from_yaml(yaml_str=basic_sample_fn.read_text())
c.plot()

# %% [markdown]
# You can see that since `gap` is defined in our cross-sections, the bundle router also intelligently picks the appropriate bundle spacing for the cross section used.
#
# Notice how the strip waveguide bundles are much more tightly packed than the rib waveguide bundles in the example below.

# %%
basic_sample_fn2 = sample_dir / "aar_bundles03.pic.yml"
show_yaml_pic(basic_sample_fn2)

# %%
f = cell_from_yaml_template(basic_sample_fn2, name="sample_transition")
c = f()
c.plot()

# %%
