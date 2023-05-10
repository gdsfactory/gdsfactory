# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Netlist extractor YAML
#
# Any component can extract its netlist with `get_netlist`
#
# While `gf.read.from_yaml` converts a `YAML Dict` into a `Component`
#
# `get_netlist` converts `Component` into a YAML `Dict`

# %% vscode={"languageId": "python"}
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %% vscode={"languageId": "python"}
c = gf.components.ring_single()
c

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
n = c.get_netlist()

# %% vscode={"languageId": "python"}
c.write_netlist("ring.yml")

# %% vscode={"languageId": "python"}
n = OmegaConf.load("ring.yml")

# %% vscode={"languageId": "python"}
i = list(n["instances"].keys())
i

# %% vscode={"languageId": "python"}
instance_name0 = i[0]

# %% vscode={"languageId": "python"}
n["instances"][instance_name0]["settings"]


# %% [markdown]
# ## Instance names
#
# By default get netlist names each `instance` with the name of the `reference`


# %% vscode={"languageId": "python"}
@gf.cell
def mzi_with_bend_automatic_naming():
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler())
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend_automatic_naming()
c.plot_netlist()


# %% vscode={"languageId": "python"}
@gf.cell
def mzi_with_bend_deterministic_names_using_alias():
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi(), alias="my_mzi")
    bend = c.add_ref(gf.components.bend_euler(), alias="my_bend")
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend_deterministic_names_using_alias()
c.plot_netlist()

# %% vscode={"languageId": "python"}
c = gf.components.mzi()
c

# %% vscode={"languageId": "python"}
c = gf.components.mzi()
n = c.get_netlist()
print(c.get_netlist().keys())

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
n.keys()


# %% [markdown]
# ## warnings
#
# Lets make a connectivity **error**, for example connecting ports on the wrong layer


# %% vscode={"languageId": "python"}
@gf.cell
def mmi_with_bend():
    c = gf.Component()
    mmi = c.add_ref(gf.components.mmi1x2(), alias="mmi")
    bend = c.add_ref(gf.components.bend_euler(layer=(2, 0)), alias="bend")
    bend.connect("o1", mmi.ports["o2"])
    return c


c = mmi_with_bend()
c

# %% vscode={"languageId": "python"}
n = c.get_netlist()

# %% vscode={"languageId": "python"}
print(n["warnings"])

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% [markdown]
# ## get_netlist_recursive
#
# When you do `get_netlist()` for a component it will only show connections for the instances that belong to that component.
# So despite having a lot of connections, it will show only the meaningful connections for that component.
# For example, a ring has a ring_coupler. If you want to dig deeper, the connections that made that ring coupler are still available.
#
# `get_netlist_recursive()` returns a recursive netlist.

# %% vscode={"languageId": "python"}
c = gf.components.ring_single()
c

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
c = gf.components.ring_double()
c

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
c = gf.components.mzit()
c

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
coupler_lengths = [10, 20, 30]
coupler_gaps = [0.1, 0.2, 0.3]
delta_lengths = [10, 100]

c = gf.components.mzi_lattice(
    coupler_lengths=coupler_lengths,
    coupler_gaps=coupler_gaps,
    delta_lengths=delta_lengths,
)
c

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
coupler_lengths = [10, 20, 30, 40]
coupler_gaps = [0.1, 0.2, 0.4, 0.5]
delta_lengths = [10, 100, 200]

c = gf.components.mzi_lattice(
    coupler_lengths=coupler_lengths,
    coupler_gaps=coupler_gaps,
    delta_lengths=delta_lengths,
)
c

# %% vscode={"languageId": "python"}
n = c.get_netlist()

# %% vscode={"languageId": "python"}
c.plot_netlist()

# %% vscode={"languageId": "python"}
n_recursive = c.get_netlist_recursive()

# %% vscode={"languageId": "python"}
n_recursive.keys()

# %% [markdown]
# ## get_netlist_flat
#
# You can also flatten the recursive netlist

# %% vscode={"languageId": "python"}
flat_netlist = c.get_netlist_flat()

# %% [markdown]
# The flat netlist contains the same keys as a regular netlist:

# %% vscode={"languageId": "python"}
flat_netlist.keys()

# %% [markdown]
# However, its instances are flattened and uniquely renamed according to hierarchy:

# %% vscode={"languageId": "python"}
flat_netlist["instances"].keys()

# %% [markdown]
# Placement information is accumulated, and connections and ports are mapped, respectively, to the ports of the unique instances or the component top level ports. This can be plotted:

# %% vscode={"languageId": "python"}
c.plot_netlist_flat(with_labels=False)  # labels get cluttered

# %% [markdown]
# ## allow_multiple_connections
#
# The default `get_netlist` function (also used by default by `get_netlist_recurse` and `get_netlist_flat`) can identify more than two ports sharing the same connection through the `allow_multiple` flag.
#
# For instance, consider a resistor network with one shared node:

# %% vscode={"languageId": "python"}
vdiv = gf.Component("voltageDivider")
r1 = vdiv << gf.components.resistance_sheet()
r2 = vdiv << gf.components.resistance_sheet()
r3 = vdiv << gf.get_component(gf.components.resistance_sheet).rotate()
r4 = vdiv << gf.get_component(gf.components.resistance_sheet).rotate()

r1.connect("pad2", r2.ports["pad1"])
r3.connect("pad1", r2.ports["pad1"], preserve_orientation=True)
r4.connect("pad2", r2.ports["pad1"], preserve_orientation=True)

vdiv

# %% vscode={"languageId": "python"}
try:
    vdiv.get_netlist_flat()
except Exception as exc:
    print(exc)

# %% vscode={"languageId": "python"}
vdiv.get_netlist_flat(allow_multiple=True)
