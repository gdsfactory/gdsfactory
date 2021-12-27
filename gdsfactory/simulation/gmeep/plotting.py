import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from meep.visualization import get_2D_dimensions

from gdsfactory.simulation.modes.types import Mode


def get_domain_measurements(sim, output_plane, frequency, resolution=0):
    """
    Modified from meep/python/visualization.py plot_eps
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


def plot_xsection(sim, center=(0, 0, 0), size=(0, 2, 2)):
    """
    sim: simulation object
    """
    sim.plot2D(output_plane=mp.Volume(center=center, size=size))
    # plt.colorbar()


def get_port_eigenmode(
    sim,
    source,
    mode_monitor,
    choose_yz=False,
    y=0,
    z=0,
):
    """
    Args:
        sim: simulation object
        source: MEEP source object
        mode_monitor: MEEP mode_monitor object to inspect
        choose_yz: whether y-z samples are generated or provided
        y: y array (if choose_yz is True)
        z: z array (if choose_yz is True)

    Returns:
        Mode object compatible with /modes plugin
    """
    # Obtain source frequency
    fsrc = source.src.frequency

    # Obtain xsection
    center = mode_monitor.regions[0].center
    size = mode_monitor.regions[0].size
    output_plane = mp.Volume(center=center, size=size)

    # Get best guess for kvector
    eps_data = get_domain_measurements(
        sim, output_plane, fsrc, resolution=1 / (y[1] - y[0]) if y else 0
    )
    n = np.sqrt(np.max(eps_data))

    # Solve for the modes
    sim.init_sim()
    eigenmode = sim.get_eigenmode(
        direction=mp.X,
        where=mp.Volume(center=center, size=size),
        band_num=1,
        kpoint=mp.Vector3(fsrc * n),
        frequency=fsrc,
    )

    # The output of this function is slightly different then MPB (there is no mode_solver object)
    # Format like the mode objects in /modes
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
            E[i, j, 0, 2] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ex
            )
            E[i, j, 0, 1] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ey
            )
            E[i, j, 0, 0] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Ez
            )
            H[i, j, 0, 2] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hx
            )
            H[i, j, 0, 1] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hy
            )
            H[i, j, 0, 0] = eigenmode.amplitude(
                mp.Vector3(center.x, yy[i, j], zz[i, j]), mp.Hz
            )

    mode = Mode(
        mode_number=1,
        neff=eigenmode.kdom / fsrc,
        wavelength=1 / fsrc,
        ng=None,  # Not currently supported
        E=E,
        H=H,
        eps=eps_data,
    )
    return mode


def plot_eigenmode(sim, source, mode_monitor):
    """
    sim: simulation object
    """
    mode = get_port_eigenmode(sim, source, mode_monitor)
    mode.plot_e_all()
    plt.show()
