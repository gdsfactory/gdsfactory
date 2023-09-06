# -*- coding: utf-8 -*-
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
# # YAML Place and AutoRoute
#
# You have two options for working with gdsfactory:
#
# 1. **python flow**: you define your layout using python functions (Parametric Cells), and connect them with routing functions.
# 2. **YAML Place and AutoRoute**: you define your Component  as Place and Route in YAML. From the netlist you can simulate the Component or generate the layout.
#
#
# YAML is a human readable version of JSON that you can use to define placements and routes
#
# to define a a YAML Component you need to define:
#
# - instances: with each instance setting
# - placements: with X and Y
#
# And optionally:
#
# - routes: between instance ports
# - connections: to connect instance ports to other ports (without routes)
# - ports: define input and output ports for the top level Component.
#
#
# gdsfactory VSCode extension has a filewatcher for `*.pic.yml` files that will show them live in klayout as you edit them.
#
# ![extension](https://i.imgur.com/89OPCQ1.png)
#
# The extension provides you with useful code snippets and filewatcher extension to see live modifications of `*pic.yml` or `*.py` files. Look for the telescope button on the top right of VSCode 🔭.
#

# %%
from functools import partial
import gdsfactory as gf
from IPython.display import Code

filepath = "yaml_pics/pads.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
gf.read.from_yaml(filepath).plot()

# %% [markdown]
# Lets start by defining the `instances` and `placements` section in YAML
#
# Lets place an `mmi_long` where you can place the `o1` port at `x=20, y=10`

# %%
filepath = "yaml_pics/mmis.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# ## ports
#
# You can expose any ports of any instance to the new Component with a `ports` section in YAML
#
# Lets expose all the ports from `mmi_long` into the new component.
#
# Ports are exposed as `new_port_name: instance_name, port_name`

# %%
filepath = "yaml_pics/ports_demo.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# You can also define a mirror placement using a port
#
# Try mirroring with other ports `o2`, `o3` or with a number as well as with a rotation `90`, `180`, `270`

# %%
filepath = "yaml_pics/mirror_demo.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# ## connections
#
# You can connect any two instances by defining a `connections` section in the YAML file.
#
# it follows the syntax `instance_source,port : instance_destination,port`

# %%
# %%
filepath = "yaml_pics/connections_demo.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# **Relative port placing**
#
# You can also place a component with respect to another instance port
#
# You can also define an x and y offset with `dx` and `dy`

# %%
filepath = "yaml_pics/relative_port_placing.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# ## routes
#
# You can define routes between two instances by defining a `routes` section in YAML
#
# it follows the syntax
#
# ```YAML
#
# routes:
#     route_name:
#         links:
#             instance_source,port: instance_destination,port
#         settings:  # for the route (optional)
#             waveguide: strip
#             width: 1.2
#
# ```

# %%
filepath = "yaml_pics/routes.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# ## instances, placements, connections, ports, routes
#
# Lets combine all you learned so far.
#
# You can define the netlist connections of a component by a netlist in YAML format
#
# Note that you define the connections as `instance_source.port ->
# instance_destination.port` so the order is important and therefore you can only
# change the position of the `instance_destination`

# %% [markdown]
# You can define several routes that will be connected using `gf.routing.get_bundle`

# %%
filepath = "yaml_pics/routes_mmi.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# You can also add custom component_factories to `gf.read.from_yaml`
#


# %%
@gf.cell
def pad_new(size=(100, 100), layer=(1, 0)):
    c = gf.Component()
    compass = c << gf.components.compass(size=size, layer=layer)
    c.ports = compass.ports
    return c


gf.get_active_pdk().register_cells(pad_new=pad_new)
c = pad_new()
f = c.plot()

# %%
filepath = "yaml_pics/pad_new.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %%
filepath = "yaml_pics/routes_custom.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()


# %% [markdown]
# Also, you can define route bundles with different settings and specify the route `factory` as a parameter as well as the `settings` for that particular route alias.

# %%
filepath = "yaml_pics/pads_path_length_match.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %%
filepath = "yaml_pics/routes_path_length_match.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %%
filepath = "yaml_pics/routes_waypoints.pic.yml"
Code(filepath, language="yaml+jinja")

# %%
c = gf.read.from_yaml(filepath)
c.plot()

# %% [markdown]
# ## Jinja Pcells
#
# You use jinja templates in YAML cells to define Pcells.

# %%
from IPython.display import Code

from gdsfactory.read import cell_from_yaml_template

gf.clear_cache()

jinja_yaml = """
default_settings:
    length_mmi:
      value: 10
      description: "The length of the long MMI"
    width_mmi:
      value: 5
      description: "The width of both MMIs"

instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: {{ width_mmi }}
        length_mmi: {{ length_mmi }}
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: {{ width_mmi }}
        length_mmi: 5
connections:
    mmi_long,o2: mmi_short,o1

ports:
    o1: mmi_long,o1
    o2: mmi_short,o2
    o3: mmi_short,o3
"""
pic_filename = "demo_jinja.pic.yml"

with open(pic_filename, mode="w") as f:
    f.write(jinja_yaml)

pic_cell = cell_from_yaml_template(pic_filename, name="demo_jinja")
gf.get_active_pdk().register_cells(
    demo_jinja=pic_cell
)  # let's register this cell so we can use it later
Code(filename=pic_filename, language="yaml+jinja")

# %% [markdown]
# You'll see that this generated a python function, with a real signature, default arguments, docstring and all!

# %%
help(pic_cell)

# %% [markdown]
# You can invoke this cell without arguments to see the default implementation

# %%
pic_cell()

# %% [markdown]
# Or you can provide arguments explicitly, like a normal cell. Note however that yaml-based cells **only accept keyword arguments**, since yaml dictionaries are inherently unordered.

# %%
pic_cell(length_mmi=100)

# %% [markdown]
# The power of jinja-templated cells become more apparent with more complex cells, like the following.

# %%
gf.clear_cache()

jinja_yaml = """
default_settings:
    length_mmis:
      value: [10, 20, 30, 100]
      description: "An array of mmi lengths for the DOE"
    spacing_mmi:
      value: 50
      description: "The vertical spacing between adjacent MMIs"
    mmi_component:
      value: mmi1x2
      description: "The mmi component to use"

instances:
{% for i in range(length_mmis|length)%}
    mmi_{{ i }}:
      component: {{ mmi_component }}
      settings:
        width_mmi: 4.5
        length_mmi: {{ length_mmis[i] }}
{% endfor %}

placements:
{% for i in range(1, length_mmis|length)%}
    mmi_{{ i }}:
      port: o1
      x: mmi_0,o1
      y: mmi_0,o1
      dy: {{ spacing_mmi * i }}
{% endfor %}

routes:
{% for i in range(1, length_mmis|length)%}
    r{{ i }}:
      routing_strategy: get_bundle_all_angle
      links:
        mmi_{{ i-1 }},o2: mmi_{{ i }},o1
{% endfor %}

ports:
{% for i in range(length_mmis|length)%}
    o{{ i }}: mmi_{{ i }},o3
{% endfor %}
"""
pic_filename = "demo_jinja_loops.pic.yml"

with open(pic_filename, mode="w") as f:
    f.write(jinja_yaml)

big_cell = cell_from_yaml_template(pic_filename, name="demo_jinja_loops")
Code(filename=pic_filename, language="yaml+jinja")

# %%
bc = big_cell()
bc.plot()

# %%
bc2 = big_cell(
    length_mmis=[10, 20, 40, 100, 200, 150, 10, 40],
    spacing_mmi=60,
    mmi_component="demo_jinja",
)
bc2.plot()

# %% [markdown]
# In general, the jinja-yaml parser has a superset of the functionalities and syntax of the standard yaml parser. The one notable exception is with `settings`. When reading any yaml files with `settings` blocks, the default settings will be read and applied, but they will not be settable, as the jinja parser has a different mechanism for setting injection with the `default_settings` block and jinja2.

# %%
pic_filename = "demo_backwards_compatibility.pic.yml"

with open(pic_filename, mode="w") as f:
    f.write(x.value)

retro_cell = cell_from_yaml_template(
    pic_filename, name="demo_jinja_backwards_compatible"
)
Code(filename=pic_filename, language="yaml")

# %%
retro_cell()  # this is fine-- because cell_from_yaml_template internally calls from_yaml, cells should work from their default state
# retro_cell(length_mmi=15) # this fails-- you must use "default_settings" and jinja syntax with the yaml-jinja parser for settings to be settable
