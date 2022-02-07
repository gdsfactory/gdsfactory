import re

import numpy as np
import pandas as pd
import tidy3d as td
from tqdm import tqdm

from gdsfactory.simulation.gtidy3d.get_results import get_results
from gdsfactory.simulation.gtidy3d.get_simulation import get_simulation
from gdsfactory.types import Component, Optional, PortSymmetries


def parse_port_eigenmode_coeff(port_index: int, ports, sim_data: td.SimulationData):
    """Given a port and eigenmode coefficient result, returns the coefficients
    relative to whether the wavevector is entering or exiting simulation

    Args:
        port_index: index of port
        ports: component_ref.ports
        sim_data:
    """
    if f"o{port_index}" not in ports:
        raise ValueError(
            f"port = 'o{port_index}' not in {list(ports.keys())}. "
            "You can rename ports with Component.auto_rename_ports()"
        )

    # Direction of port (pointing away from the simulation)
    # Figure out if that is exiting the simulation or not
    # depending on the port orientation (assuming it's near PMLs)
    if ports[f"o{port_index}"].orientation == 0:  # east
        direction_inp = "-"
        direction_out = "+"
    elif ports[f"o{port_index}"].orientation == 90:  # north
        direction_inp = "-"
        direction_out = "+"
    elif ports[f"o{port_index}"].orientation == 180:  # west
        direction_inp = "+"
        direction_out = "-"
    elif ports[f"o{port_index}"].orientation == 270:  # south
        direction_inp = "+"
        direction_out = "-"
    else:
        ValueError(
            "Port orientation {ports[port_index].orientation} is not 0, 90, 180, or 270 degrees!"
        )

    coeff_inp = sim_data.monitor_data[f"o{port_index}"].amps.sel(
        direction=direction_inp
    )
    coeff_out = sim_data.monitor_data[f"o{port_index}"].amps.sel(
        direction=direction_out
    )
    return coeff_inp, coeff_out


def get_sparameters(
    component: Component, port_symmetries: Optional[PortSymmetries] = None, **kwargs
) -> pd.DataFrame:
    """
    Get full sparameter matrix
    Simulates each time using a different input port (by default, all of them)
    unless you specify port_symmetries:

    port_symmetries = {"o1":
            {
                "s11": ["s22","s33","s44"],
                "s21": ["s21","s34","s43"],
                "s31": ["s13","s24","s42"],
                "s41": ["s14","s23","s32"],
            }
        }
    - Only simulations using the outer key port names will be run
    - The associated value is another dict whose keys are the S-parameters computed
        when this source is active
    - The values of this inner Dict are lists of s-parameters whose values are copied

    Args:
        component: to simulate.
        port_symmetries: Dict to specify port symmetries, to save number of simulations

    Keyword Args:
        mode_index: mode index
        n_modes: number of modes
        port_extension: extend ports beyond the PML
        layer_stack: contains layer numbers (int, int) to thickness, zmin
        zmargin: thickness for cladding above and below core
        thickness_pml: PML thickness (um)
        clad_material: material for cladding
        port_source_name: input port name
        port_margin: margin on each side of the port
        distance_source_to_monitors: in (um) source goes before monitors
        resolution: grid_size=3*[1/resolution]
        wavelength: in (um)
        plot_modes: plot source modes.

    """

    port_symmetries = port_symmetries or {}
    monitor_indices = []
    source_indices = []
    component_ref = component.ref()

    for port_name in component_ref.ports.keys():
        if component_ref.ports[port_name].port_type == "optical":
            monitor_indices.append(re.findall("[0-9]+", port_name)[0])
    if bool(port_symmetries):  # user-specified
        for port_name in port_symmetries.keys():
            source_indices.append(re.findall("[0-9]+", port_name)[0])
    else:  # otherwise cycle through all
        source_indices = monitor_indices

    num_sims = len(port_symmetries.keys()) or len(source_indices)
    sp = {}

    def get_sparameter(
        n: int,
        component: Component,
        port_symmetries=port_symmetries,
        monitor_indices=monitor_indices,
        **kwargs,
    ) -> np.ndarray:
        """Return Component sparameter for a particular port Index n

        Args:
            n: port_index
            component:
            port_symmetries:
            monitor_indices:

        """
        sim = get_simulation(
            component, port_source_name=f"o{monitor_indices[n]}", **kwargs
        )
        sim_data = get_results(sim).result()
        source_entering, source_exiting = parse_port_eigenmode_coeff(
            monitor_indices[n], component_ref.ports, sim_data
        )

        for monitor_index in monitor_indices:
            j = monitor_indices[n]
            i = monitor_index
            if monitor_index == monitor_indices[n]:
                sii = source_exiting / source_entering

                siia = np.unwrap(np.angle(sii))
                siim = np.abs(sii)

                sp[f"s{i}{i}a"] = siia
                sp[f"s{i}{i}m"] = siim
            else:
                monitor_entering, monitor_exiting = parse_port_eigenmode_coeff(
                    monitor_index, component_ref.ports, sim_data
                )
                sij = monitor_exiting / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)
                sp[f"s{i}{j}a"] = sija
                sp[f"s{i}{j}m"] = sijm
                sij = monitor_entering / source_entering
                sija = np.unwrap(np.angle(sij))
                sijm = np.abs(sij)

        if bool(port_symmetries) is True:
            for key in port_symmetries[f"o{monitor_indices[n]}"].keys():
                values = port_symmetries[f"o{monitor_indices[n]}"][key]
                for value in values:
                    sp[f"{value}m"] = sp[f"{key}m"]
                    sp[f"{value}a"] = sp[f"{key}a"]

        return sp

    for n in tqdm(range(num_sims)):
        sp.update(
            get_sparameter(
                n,
                component=component,
                port_symmetries=port_symmetries,
                monitor_indices=monitor_indices,
                **kwargs,
            )
        )
    return sp


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight(length=2)
    s = get_sparameters(c)
    print(s)
