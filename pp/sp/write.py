""" from a gdsfactory component write Sparameters from an FDTD Lumerical simulation
"""

import json

from collections import namedtuple
import numpy as np
import pp
from pp.layers import layer2material, layer2nm
from pp.config import materials


def write(
    component,
    session=None,
    run=True,
    overwrite=False,
    dirpath=pp.CONFIG["sp"],
    height_nm=220,
    **settings,
):
    """
    writes Sparameters from a gdsfactory component using Lumerical FDTD

    Args:
        component: gdsfactory Component
        session: you can pass a session=lumapi.FDTD() for debugging
        run: True-> runs Lumerical , False -> only draws simulation
        overwrite: run even if simulation results already exists
        dirpath: where to store the simulations
        height_nm: height
        layer2nm: dict of {(1, 0): 220}
        layer2material: dict of {(1, 0): "si"}
        remove_layers: list of tuples (layers to remove)
        background_material: for the background
        port_width: port width (m)
        port_height: port height (m)
        port_extension_um: port extension (um)
        mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine)
        zmargin: for the FDTD region 1e-6 (m)
        ymargin: for the FDTD region 2e-6 (m)
        wavelength_start: 1.2e-6 (m)
        wavelength_stop: 1.6e-6 (m)
        wavelength_points: 500

    Return:
        results: dict(wavelength_nm, S11, S12 ...) after simulation, or if simulation exists and returns the Sparameters directly
    """
    if hasattr(component, "simulation_settings"):
        settings.update(component.simulation_settings)
    sim_settings = s = dict(
        layer2nm=layer2nm,
        layer2material=layer2material,
        remove_layers=[pp.LAYER.WGCLAD],
        background_material="sio2",
        port_width=3e-6,
        port_height=1.5e-6,
        port_extension_um=1,
        mesh_accuracy=2,
        zmargin=1e-6,
        ymargin=2e-6,
        wavelength_start=1.2e-6,
        wavelength_stop=1.6e-6,
        wavelength_points=500,
    )
    for setting in settings.keys():
        assert (
            setting in s
        ), f"`{setting}` is not a valid setting ({list(settings.keys())})"
    s.update(**settings)
    ss = namedtuple("sim_settings", s.keys())(*s.values())

    assert ss.port_width < 5e-6
    assert ss.port_height < 5e-6
    assert ss.zmargin < 5e-6
    assert ss.ymargin < 5e-6

    ports = component.ports

    component.remove_layers(ss.remove_layers)
    component._bb_valid = False

    c = pp.extend_ports(component=component, length=ss.port_extension_um)
    gdspath = pp.write_gds(c)

    filepath = component.get_sparameters_path(dirpath=dirpath, height_nm=height_nm)
    filepath_json = filepath.with_suffix(".json")
    filepath_sim_settings = filepath.with_suffix(".settings.json")
    filepath_fsp = filepath.with_suffix(".fsp")

    if run and filepath_json.exists() and not overwrite:
        return json.loads(open(filepath_json).read())

    if not run and session is None:
        print(
            """
you need to pass `run=True` flag to run the simulation
To debug, you can create a lumerical FDTD session and pass it to the simulator

```
import lumapi
s = lumapi.FDTD()

import pp
c = pp.c.waveguide() # or whatever you want to simulate
pp.sp.write(component=c, run=False, session=s)
```
"""
        )

    pe = ss.port_extension_um * 1e-6 / 2
    x_min = c.xmin * 1e-6 + pe
    x_max = c.xmax * 1e-6 - pe

    y_min = c.ymin * 1e-6 - ss.ymargin
    y_max = c.ymax * 1e-6 + ss.ymargin

    port_orientations = [p.orientation for p in ports.values()]
    if 90 in port_orientations and len(ports) > 2:
        y_max = c.ymax * 1e-6 - pe
        x_max = c.xmax * 1e-6

    elif 90 in port_orientations:
        y_max = c.ymax * 1e-6 - pe
        x_max = c.xmax * 1e-6 + ss.ymargin

    z = 0
    z_span = 2 * ss.zmargin + max(ss.layer2nm.values()) * 1e-9

    import lumapi

    s = session or lumapi.FDTD(hide=False)
    s.newproject()
    s.selectall()
    s.deleteall()
    s.addrect(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z=z,
        z_span=z_span,
        index=1.5,
        name="clad",
    )

    material = ss.background_material
    if material not in materials:
        raise ValueError(f"{material} not in {list(materials.keys())}")
    material = materials[material]
    s.setnamed("clad", "material", material)

    s.addfdtd(
        dimension="3D",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z=z,
        z_span=z_span,
        mesh_accuracy=ss.mesh_accuracy,
        use_early_shutoff=True,
    )

    layers = component.get_layers()
    for layer, nm in ss.layer2nm.items():
        if layer not in layers:
            continue
        assert layer in ss.layer2material, f"{layer} not in {ss.layer2material.keys()}"

        material = ss.layer2material[layer]
        if material not in materials:
            raise ValueError(f"{material} not in {list(materials.keys())}")
        material = materials[material]

        s.gdsimport(str(gdspath), c.name, f"{layer[0]}:{layer[1]}")
        silicon = f"GDS_LAYER_{layer[0]}:{layer[1]}"
        s.setnamed(silicon, "z span", nm * 1e-9)
        s.setnamed(silicon, "material", material)

    for i, port in enumerate(ports.values()):
        s.addport()
        p = f"FDTD::ports::port {i+1}"
        s.setnamed(p, "x", port.x * 1e-6)
        s.setnamed(p, "y", port.y * 1e-6)
        s.setnamed(p, "z span", ss.port_height)

        deg = int(port.orientation)
        # assert port.orientation in [0, 90, 180, 270], f"{port.orientation} needs to be [0, 90, 180, 270]"
        if -45 <= deg <= 45:
            direction = "Backward"
            injection_axis = "x-axis"
            dxp = 0
            dyp = ss.port_width
        elif 45 < deg < 90 + 45:
            direction = "Backward"
            injection_axis = "y-axis"
            dxp = ss.port_width
            dyp = 0
        elif 90 + 45 < deg < 180 + 45:
            direction = "Forward"
            injection_axis = "x-axis"
            dxp = 0
            dyp = ss.port_width
        elif 180 + 45 < deg < -45:
            direction = "Forward"
            injection_axis = "y-axis"
            dxp = ss.port_width
            dyp = 0

        else:
            raise ValueError(
                f"port {port.name} with orientation {port.orientation} is not a valid"
                " number "
            )

        s.setnamed(p, "direction", direction)
        s.setnamed(p, "injection axis", injection_axis)
        s.setnamed(p, "y span", dyp)
        s.setnamed(p, "x span", dxp)
        # s.setnamed(p, "theta", deg)
        s.setnamed(p, "name", port.name)

    s.setglobalsource("wavelength start", ss.wavelength_start)
    s.setglobalsource("wavelength stop", ss.wavelength_stop)
    s.setnamed("FDTD::ports", "monitor frequency points", ss.wavelength_points)

    if run:
        s.save(str(filepath_fsp))
        s.deletesweep("s-parameter sweep")

        s.addsweep(3)
        s.setsweep("s-parameter sweep", "Excite all ports", 0)
        s.setsweep("S sweep", "auto symmetry", True)
        s.runsweep("s-parameter sweep")

        # collect results
        # S_matrix = s.getsweepresult("s-parameter sweep", "S matrix")
        sp = s.getsweepresult("s-parameter sweep", "S parameters")

        # export S-parameter data to file named s_params.dat to be loaded in INTERCONNECT
        s.exportsweep("s-parameter sweep", str(filepath))
        print(f"wrote sparameters to {filepath}")

        keys = [key for key in sp.keys() if key.startswith("S")]
        ra = {f"{key}a": list(np.unwrap(np.angle(sp[key].flatten()))) for key in keys}
        rm = {f"{key}m": list(np.abs(sp[key].flatten())) for key in keys}

        results = {"wavelength_nm": list(sp["lambda"].flatten() * 1e9)}
        results.update(ra)
        results.update(rm)
        with open(filepath_json, "w") as f:
            json.dump(results, f)

        with open(filepath_sim_settings, "w") as f:
            s = sim_settings
            s["layer2nm"] = [f"{k[0]}_{k[1]}_{v}" for k, v in s["layer2nm"].items()]
            s["layer2material"] = [
                f"{k[0]}_{k[1]}_{v}" for k, v in s["layer2material"].items()
            ]
            json.dump(s, f)

        return results


def write_coupler_ring():
    [
        write(
            pp.c.coupler_ring(
                wg_width=wg_width, length_x=length_x, bend_radius=bend_radius, gap=gap
            )
            for wg_width in [0.5]
            for length_x in [0.1, 1, 2, 3, 4]
            for gap in [0.15, 0.2]
            for bend_radius in [5, 10]
        )
    ]


if __name__ == "__main__":
    c = pp.c.coupler_ring(length_x=3)
    r = write(component=c)
    print(r)
    # print(r.keys())
    # print(c.ports.keys())
