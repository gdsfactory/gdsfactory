import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from gdsfactory.components import straight
from gdsfactory.simulation.gmeep import get_simulation
from gdsfactory.simulation.modes.types import Mode

'''

def get_domain_measurements(sim, output_plane, frequency, resolution=0):
    """
    Modified from meep/python/visualization.py plot_eps
    CURRENTLY UNUSED -- will be useful once the MEEP conda packages are updates to latest source
    Could also modify the epsilon plotting of mode to be override by plot_xsection, which already works
    """
    # Get domain measurements
    sim_center, sim_size = get_2D_dimensions(sim, output_plane)

    xmin = sim_center.x - sim_size.x / 2
    xmax = sim_center.x + sim_size.x / 2
    ymin = sim_center.y - sim_size.y / 2
    ymax = sim_center.y + sim_size.y / 2
    zmin = sim_center.z - sim_size.z / 2
    zmax = sim_center.z + sim_size.z / 2

    grid_resolution = resolution if resolution else sim.resolution
    Nx = int((xmax - xmin) * grid_resolution + 1)
    Ny = int((ymax - ymin) * grid_resolution + 1)
    Nz = int((zmax - zmin) * grid_resolution + 1)

    if sim_size.x == 0:
        # Plot y on x axis, z on y axis (YZ plane)
        xtics = np.array([sim_center.x])
        ytics = np.linspace(ymin, ymax, Ny)
        ztics = np.linspace(zmin, zmax, Nz)
    elif sim_size.y == 0:
        # Plot x on x axis, z on y axis (XZ plane)
        xtics = np.linspace(xmin, xmax, Nx)
        ytics = np.array([sim_center.y])
        ztics = np.linspace(zmin, zmax, Nz)
    elif sim_size.z == 0:
        # Plot x on x axis, y on y axis (XY plane)
        xtics = np.linspace(xmin, xmax, Nx)
        ytics = np.linspace(ymin, ymax, Ny)
        ztics = np.array([sim_center.z])
    else:
        raise ValueError("A 2D plane has not been specified...")

    eps_data = np.rot90(np.real(sim.get_epsilon_grid(xtics, ytics, ztics, frequency)))
    return eps_data
'''


def get_port_2Dx_eigenmode(
    sim_dict,
    source_index=0,
    port_name="o1",
    band_num=1,
    choose_yz=False,
    y=0,
    z=0,
):
    """NOTE: currently only handles ports normal to x-direction.

    Args:
        sim_dict: simulation dict
        source_index: source index (to pull from sim_dict)
        port_name: port name corresponding to mode_monitor to inspect
        band_num: band number to solve for
        choose_yz: whether y-z samples are generated or provided
        y: y array (if choose_yz is True)
        z: z array (if choose_yz is True)

    Returns:
        Mode object compatible with /modes plugin
    """
    # Initialize
    sim = sim_dict["sim"]
    source = sim_dict["sources"][source_index]
    mode_monitor = sim_dict["monitors"][port_name]

    # Obtain source frequency
    fsrc = source.src.frequency

    # Obtain xsection
    center = mode_monitor.regions[0].center
    size = mode_monitor.regions[0].size

    """
    CURRENTLY UNUSED -- will be useful once the MEEP conda packages are updates to latest source
    # output_plane = mp.Volume(center=center, size=size)
    # Get best guess for kvector
    # eps_data = get_domain_measurements(
    #     sim, output_plane, fsrc, resolution=1 / (y[1] - y[0]) if y else 0
    # )
    # n = np.sqrt(np.max(eps_data))
    """

    # Solve for the modes
    if sim_dict["initialized"] is False:
        sim.init_sim()
        sim_dict["initialized"] = True

    eigenmode = sim.get_eigenmode(
        direction=mp.X,
        where=mp.Volume(center=center, size=size),
        band_num=band_num,
        kpoint=mp.Vector3(
            fsrc * 3.45, 0, 0
        ),  # Hardcoded index for now, pull from simulation eventually
        frequency=fsrc,
    )

    # The output of this function is slightly different then MPB (there is no mode_solver object)
    # Format like the Mode objects in gdsfactory/simulation/modes to reuse modes' functions
    if not choose_yz:
        ny = int(size.y * sim.resolution)
        nz = int(size.z * sim.resolution)
        y = np.linspace(
            center.y - size.y / 2, center.y + size.y / 2, ny
        )  # eigenmode solver and sim res are technically different
        z = np.linspace(center.z - size.z / 2, center.z + size.z / 2, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    E = np.zeros([ny, nz, 1, 3], dtype=np.cdouble)
    H = np.zeros([ny, nz, 1, 3], dtype=np.cdouble)
    for i in range(ny):
        for j in range(nz):
            E[i, j, 0, 0] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ex
            )
            E[i, j, 0, 1] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ey
            )
            E[i, j, 0, 2] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ez
            )
            H[i, j, 0, 0] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hx
            )
            H[i, j, 0, 1] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hy
            )
            H[i, j, 0, 2] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hz
            )

    return Mode(
        mode_number=band_num,
        neff=eigenmode.k.x / fsrc,
        wavelength=1 / fsrc,
        ng=None,  # Not currently supported
        E=E,
        H=H,
        eps=None,  # Eventually return the index distribution for co-plotting
        y=y,
        z=z,
    )


if __name__ == "__main__":
    c = straight(length=2, width=0.5)
    c = c.copy()
    c.add_padding(default=0, bottom=3, top=3, layers=[(100, 0)])

    sim_dict = get_simulation(
        c,
        is_3d=True,
        res=50,
        port_source_offset=-0.1,
        port_field_monitor_offset=-0.1,
        port_margin=2.5,
    )

    m1_MEEP = get_port_2Dx_eigenmode(
        sim_dict=sim_dict,
        source_index=0,
        port_name="o1",
    )
    print(m1_MEEP.neff)
    m1_MEEP.plot_hy()
    m1_MEEP.plot_hx()
    m1_MEEP.plot_hz()
    plt.show()
