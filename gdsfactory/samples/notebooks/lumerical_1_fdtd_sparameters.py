# # FDTD Lumerical
#
# gdsfactory provides you with a Lumerical FDTD interface to calculate Sparameters automatically  (without you having to click around the Lumerical GUI)
#
# The function `gdsfactory.simulation.lumerical.write_sparameters_lumerical` brings up a GUI, runs simulation and then writes the Sparameters both in .CSV and .DAT (Lumerical interconnect / simphony) file formats, as well as the simulation settings in YAML format.
#
# In the CSV format each Sparameter will have 2 columns, `o1@0,o2@0` where `m` stands for magnitude and `s12a` where `a` stands for angle in radians.
#
# For the simulation to wor well, your components need to have ports, that will be extended automatically to go over the PML.
#
# ![lum GUI](https://i.imgur.com/dHAzZRw.png)
#
#
# The script calls internally the lumerical python API `lumapi` so you will need to make sure that you can run this from python.
#
# ```python
# import lumapi
#
# session = lumapi.FDTD()
# ```
#
# In linux that may require you to export the PYTHONPATH variable in your shell environment.
#
# You can add one line into your `.bashrc` in your Linux machine.
#
#
# ```bash
# [ -d "/opt/lumerical" ] && export PATH=$PATH:/opt/lumerical/$(ls /opt/lumerical)/bin && export PYTHONPATH=/opt/lumerical/$(ls /opt/lumerical)/api/python
# ```
#
#
# Finally, You can chain the Sparameters to calculate solve of larger circuits using a circuit solver such as
#
# - Lumerical interconnect
# - [sax (open source)](https://sax.readthedocs.io/en/latest/index.html)
#

# +
from gdsfactory.generic_tech import LAYER_STACK
import gdsfactory.simulation.lumerical as sim
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

gf.technology.SIMULATION_SETTINGS_LUMERICAL_FDTD

# +
# sim.write_sparameters_lumerical?

# +
import lumapi

s = lumapi.FDTD()
# -

gf.components.cells.keys()

# +
components = [
    "bend_euler",
    "bend_s",
    "coupler",
    "coupler_ring",
    "crossing",
    "mmi1x2",
    "mmi2x2",
    "taper",
    "straight",
]
need_review = []

for component_name in components:
    component = gf.components.cells[component_name]()
    sim.write_sparameters_lumerical(component, run=False, session=s)
    response = input(f"does the simulation for {component_name} look good? (y/n)")
    if response.upper()[0] == "N":
        need_review.append(component_name)
# -

# ## Modify layer stack
#
# All layers information is passed to the Lumerical simulator through the layer_stack
#
#
# ### Layer thickness
#
# You can modify the thickness of any specific layer of the stack. For example lets increase the core thickness to `230 nm`

LAYER_STACK

layer_stack2 = LAYER_STACK

nm = 1e-3
layer_stack2["core"].thickness = 230 * nm

layer_stack2["core"].thickness

sim.write_sparameters_lumerical(
    gf.components.mmi1x2(), layer_stack=layer_stack2, run=False, session=s
)

# You will be able to see the layer thickness increase in the lumerical GUI
#
# ![thickness](https://i.imgur.com/Hxe7BuC.png)

# ### Layer material or index
#
# You can also modify the material refractive index or material name from the Lumerical Material database
#
#
# material: material spec, can be
#
# -  a string from lumerical database materials.
# -  a complex for n, k materials.
# -  a float or int, representing refractive index.
#

LAYER_STACK

# +
material_name_to_lumerical = dict(si=3.6)

sim.write_sparameters_lumerical(
    gf.components.mmi1x2(),
    layer_stack=layer_stack2,
    run=False,
    session=s,
    material_name_to_lumerical=material_name_to_lumerical,
)
# -

# ![stack](https://i.imgur.com/ywfnH6h.png)

# +
component = gf.components.mmi1x2()
material_name_to_lumerical = dict(si=(3.45, 2))  # or dict(si=3.45+2j)

r = sim.write_sparameters_lumerical(
    component=component,
    material_name_to_lumerical=material_name_to_lumerical,
    run=False,
    session=s,
)
# -

# ![complex index](https://i.imgur.com/Tbv1Mbb.png)

# +
material_name_to_lumerical = dict(si="InP - Palik")

sim.write_sparameters_lumerical(
    gf.components.mmi1x2(),
    layer_stack=layer_stack2,
    run=False,
    session=s,
    material_name_to_lumerical=material_name_to_lumerical,
)
# -

# ![extra](https://i.imgur.com/75IR6fa.png)

# gdsfactory can also compute the Sparameters of a component that have not been simulated before.

sim.write_sparameters_lumerical(gf.components.mmi1x2())

sim.plot.plot_sparameters(gf.components.mmi1x2(), keys=["S23m", "S13m"], logscale=True)

# As well as a group of components

# +
components = [
    gf.components.coupler_ring(gap=gap, radius=radius)
    for gap in [0.15, 0.2, 0.3]
    for radius in [5, 10]
]

for c in components:
    c.show(show_ports=True)
    print(c)
    sim.write_sparameters_lumerical(c)
# -

# To debug a simulation you can create a Lumerical session outside the simulator, pass it to the simulator, and use `run=False` flag

s = lumapi.FDTD()
c = gf.components.straight()
sim.write_sparameters_lumerical(c, run=False, session=s)


# By default gdsfactory uses the generic LayerStack for 0.22um height silicon layer.
#
# You can also define any LayerStack
#


# +
def get_layer_stack():
    return gf.tech.LayerStack(
        wg=gf.tech.LayerLevel(layer=(1, 0), thickness=400e-3, zmin=0.0, material="sin")
    )


layer_stack = get_layer_stack()

# +

c = gf.components.straight()
s = sim.write_sparameters_lumerical(c, layer_stack=layer_stack, run=False, session=s)
