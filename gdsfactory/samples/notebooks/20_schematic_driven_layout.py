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
# # Schematic Driven layout
#
# The Schematic driven layout uses a schematic format similar to our `*.pic.yml`.
#
# The Jupyter notebook interface allows you to get the best of both worlds of GUI and python driven based flows.

# %%
from bokeh.io import output_notebook

from gdsfactory.schematic_editor import SchematicEditor
from gdsfactory.config import rich_output
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %env BOKEH_ALLOW_WS_ORIGIN=127.0.0.1:8888,localhost:8888

output_notebook()
rich_output()

# %% [markdown]
# First you initialize a session of the schematic editor.
# The editor is synced to a file.
# If file exist, it loads the schematic for editing. If it does not exist, it creates it.
# The schematic file is continuously auto-saved as you edit the schematic in your notebook, so you can track changes with GIT.

# %%
se = SchematicEditor("test.schem.yml")

# %% [markdown]
# ## Define instances
#
# First you need to define which instances to include. We do this through this grid-like editor.
# Components are auto-populated from your active PDK.
#
# instance name | Component type
# --------------| --------------
# mmi1          | mmi1x2

# %%
se.instance_widget

# %%
se.instances.keys()

# %% [markdown]
# You can also **add your instances through code**, and since it is just a dictionary update, *the integrity of your schematic will be maintained, even after re-running the notebook* as-is.
# You can here specify a component either by name or as an actual component, using auto-completion to specify your settings as well.

# %%
se.add_instance("s1", gf.components.straight(length=20))
se.add_instance("s2", gf.components.straight(length=40))

# %% [markdown]
# But you can even query the parameters of default components, set only by name through the widget grid, like so:

# %%
se.instances["mmi1"].settings.full

# %% [markdown]
# It is also possible to *instantiate through the widget, then set the settings of our component later, through code.*
#
# By doing this through code, we have the full power of python at our disposal to easily use shared variables between components, or set complex Class or dictionary-based settings, without fumbling through a UI.

# %%
se.update_settings("mmi1", gap_mmi=1.0)
se.update_settings("mmi2", gap_mmi=0.7)

for inst_name, inst in se.instances.items():
    if inst.settings.changed:
        print(f"{inst_name}: {inst.settings.changed}")

# %% [markdown]
# ## Define nets
#
# Now, you define your logical connections between instances in your netlist. Each row in the grid represents one logical connection.

# %%
se.net_widget

# %% [markdown]
# Similarly, you can programmatically add nets.
# Adding a net which already exists will have no effect, such that the notebook can be rerun without consequence.
# However, trying to connect to a port which is already otherwise connected will throw an error.

# %%
se.add_net(
    inst1="mmi1", port1="o2", inst2="s1", port2="o1"
)  # can be re-run without consequence
se.add_net(inst1="s1", port1="o1", inst2="mmi1", port2="o2")  # also ok
# se.add_net(inst1="s1", port1="o2", inst2="mmi1", port2="o2")  # throws error -- already connected to a different port

# %%
se.schematic

# %% [markdown]
# ## Define ports
#
# Now, you define the Component ports following the syntax
#
# PortName | InstanceName,PortName

# %%
se.port_widget

# %% [markdown]
# ## Visualize
#
# You can visualize your schematic down below. After you've initialized the plot below, you should see it live-update after every change we make above.
#
# Currently the size of component symbols and port locations are **layout-realistic**.
# This may be a nice default if you don't care to bother making symbols for your components.
# But it would be a nice improvement for the future to allow associating symbols with components, to make the schematic easier to read.
#
# If you activate the `Point Draw Tool` in the plot, you should see that you are able to arrange components freely on the schematic, and changes are saved back to the `*.schem.yml` file in realtime.
#
# ![pointdraw](https://i.imgur.com/mlfsd13.png)

# %%
se.visualize()

# %% [markdown]
# ## generate Layout
#
# You can use your schematic to generate a preliminary layout, and view in the notebook and/or KLayout. Initial placements come from schematic placements and Routes are auto-generated from nets.

# %%
layout_filename = "sdl_demo.pic.yml"
se.instantiate_layout(layout_filename, default_router="get_bundle")
c = gf.read.from_yaml(layout_filename)
c.plot()

# %%
# you can save your schematic to a standalone html file once you are satisfied
# se.save_schematic_html('demo_schem.html')

# %% [markdown]
# ## Circuit simulations

# %%
import numpy as np
import matplotlib.pyplot as plt
import gdsfactory.simulation.sax as gs
import jax.numpy as jnp
import sax

netlist = c.get_netlist()

models = {
    "bend_euler": gs.models.bend,
    "mmi1x2": gs.models.mmi1x2,
    "mmi2x2": gs.models.mmi2x2,
    "straight": gs.models.straight,
}

circuit, _ = sax.circuit(netlist=netlist, models=models)

# %%
wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
