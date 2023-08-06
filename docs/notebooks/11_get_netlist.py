# # Netlist extractor YAML
#
# Any component can extract its netlist with `get_netlist`
#
# While `gf.read.from_yaml` converts a `YAML Dict` into a `Component`
#
# `get_netlist` converts `Component` into a YAML `Dict`

# +
from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()
# -

c = gf.components.mzi()
c.plot()

c.plot_netlist()

c = gf.components.ring_single()
c.plot()

c.plot_netlist()

n = c.get_netlist()

c.write_netlist("ring.yml")

n = OmegaConf.load("ring.yml")

i = list(n["instances"].keys())
i

instance_name0 = i[0]

n["instances"][instance_name0]["settings"]


# ## Instance names
#
# By default get netlist names each `instance` with the name of the `reference`
#


# +
@gf.cell
def mzi_with_bend_automatic_naming():
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler())
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend_automatic_naming()
c.plot_netlist()


# +
@gf.cell
def mzi_with_bend_deterministic_names_using_alias():
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi(), alias="my_mzi")
    bend = c.add_ref(gf.components.bend_euler(), alias="my_bend")
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend_deterministic_names_using_alias()
c.plot_netlist()
# -

c = gf.components.mzi()
c.plot()

c = gf.components.mzi()
n = c.get_netlist()
print(c.get_netlist().keys())

c.plot_netlist()

n.keys()


# ## warnings
#
# Lets make a connectivity **error**, for example connecting ports on the wrong layer
#


# +
@gf.cell
def mmi_with_bend():
    c = gf.Component()
    mmi = c.add_ref(gf.components.mmi1x2(), alias="mmi")
    bend = c.add_ref(gf.components.bend_euler(layer=(2, 0)), alias="bend")
    bend.connect("o1", mmi.ports["o2"])
    return c


c = mmi_with_bend()
c.plot()
# -

n = c.get_netlist()

print(n["warnings"])

c.plot_netlist()

# ## get_netlist_recursive
#
# When you do `get_netlist()` for a component it will only show connections for the instances that belong to that component.
# So despite having a lot of connections, it will show only the meaningful connections for that component.
# For example, a ring has a ring_coupler. If you want to dig deeper, the connections that made that ring coupler are still available.
#
# `get_netlist_recursive()` returns a recursive netlist.

c = gf.components.ring_single()
c.plot()

c.plot_netlist()

c = gf.components.ring_double()
c.plot()

c.plot_netlist()

c = gf.components.mzit()
c.plot()

c.plot_netlist()

# +
coupler_lengths = [10, 20, 30]
coupler_gaps = [0.1, 0.2, 0.3]
delta_lengths = [10, 100]

c = gf.components.mzi_lattice(
    coupler_lengths=coupler_lengths,
    coupler_gaps=coupler_gaps,
    delta_lengths=delta_lengths,
)
c.plot()
# -

c.plot_netlist()

# +
coupler_lengths = [10, 20, 30, 40]
coupler_gaps = [0.1, 0.2, 0.4, 0.5]
delta_lengths = [10, 100, 200]

c = gf.components.mzi_lattice(
    coupler_lengths=coupler_lengths,
    coupler_gaps=coupler_gaps,
    delta_lengths=delta_lengths,
)
c.plot()
# -

n = c.get_netlist()

c.plot_netlist()

n_recursive = c.get_netlist_recursive()

n_recursive.keys()

# ## get_netlist_flat
#
# You can also flatten the recursive netlist

flat_netlist = c.get_netlist_flat()

# The flat netlist contains the same keys as a regular netlist:

flat_netlist.keys()

# However, its instances are flattened and uniquely renamed according to hierarchy:

flat_netlist["instances"].keys()

# Placement information is accumulated, and connections and ports are mapped, respectively, to the ports of the unique instances or the component top level ports. This can be plotted:

c.plot_netlist_flat(with_labels=False)  # labels get cluttered

# ## allow_multiple_connections
#
# The default `get_netlist` function (also used by default by `get_netlist_recurse` and `get_netlist_flat`) can identify more than two ports sharing the same connection through the `allow_multiple` flag.
#
# For instance, consider a resistor network with one shared node:

# +
vdiv = gf.Component("voltageDivider")
r1 = vdiv << gf.components.resistance_sheet()
r2 = vdiv << gf.components.resistance_sheet()
r3 = vdiv << gf.get_component(gf.components.resistance_sheet).rotate()
r4 = vdiv << gf.get_component(gf.components.resistance_sheet).rotate()

r1.connect("pad2", r2.ports["pad1"])
r3.connect("pad1", r2.ports["pad1"], preserve_orientation=True)
r4.connect("pad2", r2.ports["pad1"], preserve_orientation=True)

vdiv
# -

try:
    vdiv.get_netlist_flat()
except Exception as exc:
    print(exc)

vdiv.get_netlist_flat(allow_multiple=True)
