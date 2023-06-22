# # Lumerical INTERCONNECT
#
# The Lumerical INTERCONNECT plugin in gdsfactory can run circuit simulations in INTERCONNECT directly from gdsfactory components.
#
# This is a work-in-progress and can't handle hierarchical components yet.
#
#
# This example also requires you to install the ubcpdk `pip install ubcpdk`

# +
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from gdsfactory.simulation.lumerical.interconnect import plot_wavelength_sweep
from gdsfactory.simulation.lumerical.interconnect import run_wavelength_sweep
from gdsfactory.get_netlist import get_instance_name_from_alias as get_instance_name
from gdsfactory.routing import get_route
import gdsfactory as gf

import ubcpdk.components as pdk

gf.config.rich_output()

# +
import lumapi

session = lumapi.INTERCONNECT()
# -

# Currently, only simulations using CMLs (compact model libraries) are supported, so this tutorial will demonstrate using the [SiEPIC EBeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK) with the [ubcpdk](https://github.com/gdsfactory/ubc) package.
#

# +
circuit = gf.Component("Circuit")

gc1 = circuit << pdk.gc_te1550()
gc2 = circuit << pdk.gc_te1550()
gc3 = circuit << pdk.gc_te1550()

gc1.rotate(180)
gc2.rotate(180)
gc3.rotate(180)

gc2.movey(127)
gc3.movey(-127)

s = circuit << pdk.y_splitter()
s.movex(75)

circuit.show()
circuit

# +
route_in = get_route(gc1.ports["opt1"], s.ports["opt1"])
route_out_top = get_route(s.ports["opt2"], gc2.ports["opt1"])
route_out_bot = get_route(
    s.ports["opt3"], gc3.ports["opt1"], start_straight_length=1000
)

circuit.add(route_in.references)
circuit.add(route_out_top.references)
circuit.add(route_out_bot.references)

circuit.show()

# +
netlist = circuit.get_netlist()

gc1_netlist_instance_name = get_instance_name(circuit, gc1)
gc2_netlist_instance_name = get_instance_name(circuit, gc2)
gc3_netlist_instance_name = get_instance_name(circuit, gc3)

ports_in = {gc1_netlist_instance_name: "opt_fiber"}
ports_out = {
    gc2_netlist_instance_name: "opt_fiber",
    gc3_netlist_instance_name: "opt_fiber",
}

# +
simulation_settings = OrderedDict(
    [
        ("MC_uniformity_thickness", np.array([200, 200])),
        ("MC_uniformity_width", np.array([200, 200])),
        ("MC_non_uniform", 0),
        ("MC_grid", 1e-5),
        ("MC_resolution_x", 200),
        ("MC_resolution_y", 0),
    ]
)

results = run_wavelength_sweep(
    component=circuit,
    session=session,
    ports_in=ports_in,
    ports_out=ports_out,
    simulation_settings=simulation_settings,
    results=("transmission",),
    component_distance_scaling=10,
    setup_mc=True,
)
# -

plot_wavelength_sweep(ports_out=ports_out, results=results, show=True)

# ## MZI Wavelength Sweep

mzi = pdk.mzi()
mzi

# +
# If the ports are in the top-level cell, use a dictionary like this and
# set is_top_level to True in the call to run_wavelength_sweep
ports_in = {"o1": "o1"}
ports_out = {"o2": "o2"}

simulation_settings = OrderedDict(
    [
        ("MC_uniformity_thickness", np.array([200, 200])),
        ("MC_uniformity_width", np.array([200, 200])),
        ("MC_non_uniform", 0),
        ("MC_grid", 1e-5),
        ("MC_resolution_x", 200),
        ("MC_resolution_y", 0),
    ]
)
results = run_wavelength_sweep(
    session=session,
    component=mzi,
    ports_in=ports_in,
    ports_out=ports_out,
    results=("transmission",),
    component_distance_scaling=50,
    simulation_settings=simulation_settings,
    setup_mc=True,
    is_top_level=True,
)
# -

plot_wavelength_sweep(
    ports_out=ports_out, results=results, result_name="'TE' transmission"
)

# +
um = 1e-6
result_name = "'TE' transmission"

for port in ports_out:
    wl = results["transmission"][port]["wavelength"] / um
    T = 10 * np.log10(np.abs(results["transmission"][port][result_name]))
    plt.plot(wl, T, label=str(port))

plt.legend()
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel(f"{result_name} (dB)")
plt.show()
