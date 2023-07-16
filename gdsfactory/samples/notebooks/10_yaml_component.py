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
# 2. **YAML Place and AutoRoute**: you define your circuit (Place and Route) in YAML. From the netlist you can simulate the circuit or generate the layout.
#
#
# YAML is a human readable version of JSON that you can use to define components placements and routes
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
# - ports: define input and output circuit ports
#
#
# When running this tutorial make sure you UNCOMMENT this line `%matplotlib widget` so you can see the changes in the YAML file both in KLayout and matplotlib.
#
# `# %matplotlib widget`  -> `%matplotlib widget`

# %%
# # %matplotlib widget

# %%
import ipywidgets
from IPython.display import display
from functools import partial

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

x = ipywidgets.Textarea(rows=20, columns=480)

x.value = """
name: sample_different_factory

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: 200
        y: 500

    br:
        x: 400
        y: 400

    tr:
        x: 400
        y: 600


routes:
    electrical:
        settings:
            separation: 20
            layer: [41, 0]
            width: 10
        links:
            tl,e3: tr,e1
            bl,e3: br,e1
    optical:
        settings:
            radius: 100
        links:
            bl,e4: br,e3
"""

out = ipywidgets.Output()
display(x, out)


def f(change, out=out):
    try:
        c = gf.read.from_yaml(change["new"])
        c.show(show_ports=True)
        c.plot_klayout()
        out.clear_output()
    except Exception as e:
        out.clear_output()
        with out:
            display(e)


x.observe(f, "value")
f({"new": x.value})

# %% [markdown]
# Lets start by defining the `instances` and `placements` section in YAML
#
# Lets place an `mmi_long` where you can place the `W0` port at `x=20, y=10`

# %%
x.value = """
name: mmis
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_long:
        port: o1
        x: 20
        y: 10
        mirror: False
"""
display(x, out)

# %%
x.value = """
name: mmi_mirror
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_long:
        port: o1
        x: 20
        y: 10
        mirror: False
"""
display(x, out)

# %% [markdown]
# ## ports
#
# You can expose any ports of any instance to the new Component with a `ports` section in YAML
#
# Lets expose all the ports from `mmi_long` into the new component.
#
# Ports are exposed as `new_port_name: instance_name, port_name`
#
# you can see the ports in `red` and subports in `blue`

# %%
x.value = """
name: ports_demo
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        port: o1
        x: 20
        y: 10
        mirror: True

ports:
    o3: mmi_long,o3
    o2: mmi_long,o2
    o1: mmi_long,o1
"""

display(x, out)

# %% [markdown]
# You can also define a mirror placement using a port
#
# Try mirroring with other ports `o2`, `o3` or with a number as well as with a rotation `90`, `180`, `270`

# %%
x.value = """
name: mirror_demo
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        x: 0
        y: 0
        mirror: o1
        rotation: 0
"""

display(x, out)

# %% [markdown]
# ## connections
#
# You can connect any two instances by defining a `connections` section in the YAML file.
#
# it follows the syntax.
#
# `instance_source,port : instance_destination,port`

# %%
x.value = """
name: connections_demo
instances:
    b:
      component: bend_circular
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_short:
        port: o1
        x: 10
        y: 20
connections:
    b,o1 : mmi_short,o2
    mmi_long,o1: b, o2

ports:
    o1: mmi_short,o1
    o2: mmi_long,o2
    o3: mmi_long,o3
"""

display(x, out)

# %% [markdown]
# **Relative port placing**
#
# You can also place a component with respect to another instance port
#
# You can also define an x and y offset with `dx` and `dy`

# %%
x.value = """
name: rel_port_placing
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5

placements:
    mmi_short:
        port: o1
        x: 0
        y: 0
    mmi_long:
        port: o1
        x: mmi_short,o2
        y: mmi_short,o2
        dx : 10
        dy: -10
"""


display(x, out)

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
x.value = """
name: with_routes
instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 10
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
placements:
    mmi_long:
        x: 100
        y: 100
routes:
    optical:
        links:
            mmi_short,o2: mmi_long,o1
        settings:
            cross_section:
                cross_section: strip
                settings:
                    layer: [2, 0]
"""

display(x, out)

# %% [markdown]
# You can **rotate** and instance specifying the angle in degrees

# %% [markdown]
# You can also access the routes in the newly created component

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
x.value = """
name: connections_2x2_problem

instances:
    mmi_bottom:
      component: mmi2x2
    mmi_top:
      component: mmi2x2

placements:
    mmi_top:
        x: 100
        y: 100

routes:
    optical:
        links:
            mmi_bottom,o4: mmi_top,o1
            mmi_bottom,o3: mmi_top,o2

"""

display(x, out)

# %% [markdown]
# You can also add custom component_factories to `gf.read.from_yaml`
#


# %%
@gf.cell
def pad_new(size=(100, 100), layer=gf.LAYER.WG):
    c = gf.Component()
    compass = c << gf.components.compass(size=size, layer=layer)
    c.ports = compass.ports
    return c


gf.get_active_pdk().register_cells(pad_new=pad_new)
c = pad_new(cache=False)
f = c.plot()

# %%
x.value = """
name: connections_2x2_problem

instances:
    bot:
      component: pad_new
    top:
      component: pad_new

placements:
    top:
        x: 0
        y: 200
"""

display(x, out)

# %%
x.value = """
name: custom_routes

instances:
    t:
      component: pad_array
      settings:
          orientation: 270
          columns: 3
    b:
      component: pad_array
      settings:
          orientation: 90
          columns: 3

placements:
    t:
        x: 200
        y: 400
routes:
    electrical:
        settings:
            layer: [31, 0]
            width: 10.
            end_straight_length: 150
        links:
            t,e11: b,e11
            t,e13: b,e13
"""

display(x, out)

# %% [markdown]
# Also, you can define route aliases, that have different settings and specify the route `factory` as a parameter as well as the `settings` for that particular route alias.

# %%
x.value = """
name: sample_settings

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: 0
        y: 200

    br:
        x: 400
        y: 400

    tr:
        x: 400
        y: 600

routes:
    optical_r100:
        settings:
            radius: 100
            layer: [31, 0]
            width: 50
        links:
            tl,e2: tr,e2
    optical_r200:
        settings:
            radius: 200
            width: 10
            layer: [31, 0]
        links:
            bl,e3: br,e3
"""

display(x, out)

# %%
x.value = """
instances:
    t:
      component: pad_array
      settings:
          orientation: 270
          columns: 3
    b:
      component: pad_array
      settings:
          orientation: 90
          columns: 3

placements:
    t:
        x: 200
        y: 500
routes:
    optical:
        settings:
            radius: 50
            width: 40
            layer: [31,0]
            end_straight_length: 150
            separation: 50
        links:
            t,e11: b,e11
            t,e12: b,e12
            t,e13: b,e13
"""

display(x, out)

# %%
x.value = """

instances:
    t:
      component: pad_array
      settings:
          orientation: 270
          columns: 3
    b:
      component: pad_array
      settings:
          orientation: 90
          columns: 3

placements:
    t:
        x: 100
        y: 1000
routes:
    route1:
        routing_strategy: get_bundle_path_length_match
        settings:
            extra_length: 500
            width: 2
            layer: [31,0]
            end_straight_length: 500
        links:
            t,e11: b,e11
            t,e12: b,e12
"""

display(x, out)

# %%
x.value = """
instances:
    t:
      component: pad_array
      settings:
          orientation: 270
          columns: 3
    b:
      component: pad_array
      settings:
          orientation: 90
          columns: 3

placements:
    t:
        x: -250
        y: 1000
routes:
    route1:
        routing_strategy: get_bundle_from_waypoints
        settings:
            waypoints:
                - [0, 300]
                - [400, 300]
                - [400, 400]
                - [-250, 400]
            auto_widen: False
        links:
            b,e11: t,e11
            b,e12: t,e12

"""

display(x, out)

# %% [markdown]
# Note that you define the connections as `instance_source.port -> instance_destination.port` so the order is important and therefore you can only change the position of the `instance_destination`

# %% [markdown]
# ## Custom factories
#
# You can leverage netlist defined components to define more complex circuits

# %%
mmi1x2_faba = partial(gf.components.mmi1x2, length_mmi=30)
mmi2x2_faba = partial(gf.components.mmi2x2, length_mmi=30)
gf.get_active_pdk().register_cells(mmi1x2_faba=mmi1x2_faba, mmi2x2_faba=mmi2x2_faba)

x.value = """
name: sample_custom_cells
instances:
    mmit:
      component: mmi2x2_faba
    mmib:
      component: mmi1x2_faba
      settings:
        width_mmi: 4.5
placements:
    mmit:
        x: 100
        y: 100
routes:
    route1:
        links:
            mmib,o2: mmit,o2

ports:
    o1: mmib,o1
    o2: mmit,o2
    o3: mmit,o3
    o4: mmit,o4
"""

display(x, out)

# %%
c = gf.components.mzi()
c.plot()

# %%
c.plot_netlist()

# %%
n = c.get_netlist()

# %%
print(c.get_netlist().keys())

# %% [markdown]
# ## variables
#
#
# You can define a global variables `settings` in your YAML file, and use the variable in the other YAML settings by using `${settings.length_mmi}`

# %%
x.value = """
settings:
    length_mmi: 10

instances:
    mmi_long:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: ${settings.length_mmi}
    mmi_short:
      component: mmi1x2
      settings:
        width_mmi: 4.5
        length_mmi: 5
"""

display(x, out)

# %% [markdown]
# ## `cell_from_yaml_template`: Jinja-template-based Parser
# An optional parser variant is also available which is capable of parsing jinja templating directives within the yaml-based cells. This can give python-like flexibility inside the otherwise declaratively-defined yaml circuit syntax.

# %%
from gdsfactory.read import cell_from_yaml_template
from IPython.display import Code

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
# pic_cell?

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
bc

# %%
bc2 = big_cell(
    length_mmis=[10, 20, 40, 100, 200, 150, 10, 40],
    spacing_mmi=60,
    mmi_component="demo_jinja",
)
bc2

# %% [markdown]
# ## Choosing your preferred yaml parser
#

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

# %% [markdown]
# Because of this incompatibility, you must choose one parser or another to be used by default at the scope of the PDK.

# %%
gf.get_active_pdk().circuit_yaml_parser = cell_from_yaml_template
