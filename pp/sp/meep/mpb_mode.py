""" Compute electromagnetic waveguide modes in MPB. Based on PICwriter.

Launches a MPB simulation to compute the electromagnetic mode profile for a given waveguide template and
material stack.
"""

import pathlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import meep as mp
from meep import mpb
from pp.components.waveguide_template import wg_strip
from pp.layers import LAYER


class MaterialStack:
    """Standard template for generating a material stack

    Args:
       * **vsize** (float): Vertical size of the material stack in microns (um)
       * **default_layer** (list): Default VStack with the following format: [(eps1, t1), (eps2, t2), (eps3, t3), ...] where eps1, eps2, .. are the permittivity (float), and t1, t2, .. are the thicknesses (float) from bottom to top. Note: t1+t2+... *must* add up to vsize.

    Members:
       * **stacklist** (dictionary): Each entry of the stacklist dictionary contains a VStack list.

    Keyword Args:
       * **name** (string): Identifier (optional) for the material stack

    """

    def __init__(self, vsize, default_stack, name="mstack"):
        self.name = name
        self.vsize = vsize
        self.default_stack = default_stack

        """ self.stack below contains a DICT of all the VStack lists """
        self.stacklist = {}

        self.addVStack(-1, -1, default_stack)

    def addVStack(self, layer, datatype, stack):
        """Adds a vertical layer to the material stack LIST

        Args:
           * **layer** (int): Layer of the VStack
           * **datatype** (int): Datatype of the VStack
           * **stack** (list): Vstack list with the following format: [(eps1, t1), (eps2, t2), (eps3, t3), ...] where eps1, eps2, .. are the permittivity (float), and t1, t2, .. are the thicknesses (float) from bottom to top. Note: if t1+t2+... must add up to vsize.

        """
        # First, typecheck the stack
        t = 0
        for s in stack:
            t += s[1]
        if abs(t - self.vsize) >= 1e-6:
            raise ValueError(
                "Warning! Stack thicknesses ("
                + str(t)
                + ") do not add up to vsize ("
                + str(self.vsize)
                + ")."
            )

        self.stacklist[(layer, datatype)] = stack

    def interpolate_points(self, key, num_points):
        layer_ranges = []
        curz = 0.0  # "Current z"
        for layer in self.stacklist[key]:
            layer_ranges.append([layer[0], curz, curz + layer[1]])
            curz = curz + layer[1]

        points = []
        for i in np.linspace(1e-8, self.vsize, num_points):
            for layer in layer_ranges:
                if i > layer[1] and i <= layer[2]:
                    points.append(layer[0])

        if len(points) is not num_points:
            raise ValueError("Point interpolation did not work.  Repeat points added.")

        return np.array(points)

    def get_eps(self, key, height):
        """Returns the dielectric constant (epsilon) corresponding to the `height` and VStack specified by `key`, where the height range is zero centered (-vsize/2.0, +vsize/2.0).

        Args:
            * **key** (layer,datatype): Key value of the VStack being used
            * **height** (float): Vertical position of the desired epsilon value (must be between -vsize/2.0, vsize/2.0)

        """
        cur_vmax = -self.vsize / 2.0
        for layer in self.stacklist[key]:
            cur_vmax += layer[1]
            if height <= cur_vmax:
                return layer[0]
        return self.stacklist[key][-1][0]


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def export_wgt_to_hdf5(filename, wgt, mstack, sx):
    """Outputs the polygons corresponding to the desired waveguide template and MaterialStack.
    Format is compatible for generating prism geometries in MEEP/MPB.

    **Note**: that the top-down view of the device is the 'X-Z' plane.  The 'Y' direction specifies the vertical height.

    Args:
       * **filename** (string): Filename to save (must end with '.h5')
       * **wgt** (WaveguideTemplate): WaveguideTemplate object from the PICwriter library
       * **mstack** (MaterialStack): MaterialStack object that maps the gds layers to a physical stack
       * **sx** (float): Size of the simulation region in the x-direction

    Write-format for all blocks:
       * CX = center-x
       * CY = center-y
       * width = width (x-direction) of block
       * height = height (y-direction) of block
       * eps = dielectric constant

    """
    CX, CY, width_list, height_list, eps_list = [], [], [], [], []
    if wgt.wg_type == "strip":
        """
        check wg (layer/datatype) and clad (layer/datatype)
        check if these are in the mstack.  If so, create the blocks that would correspond to each
        save all 'block' info to hdf5 file (center, x-size, y-size, epsilon)
        still need to add support for full material functions (i.e. built-in silicon, SiN, etc...)
        """
        for key in mstack.stacklist.keys():
            if key == (wgt.wg_layer, wgt.wg_datatype):
                width = wgt.wg_width
                center_x = 0.0
                total_y = sum([layer[1] for layer in mstack.stacklist[key]])
                cur_y = -total_y / 2.0
                for layer in mstack.stacklist[key]:
                    center_y = cur_y + layer[1] / 2.0
                    cur_y = cur_y + layer[1]
                    CX.append(center_x)
                    CY.append(center_y)
                    width_list.append(width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
            if key == (wgt.clad_layer, wgt.clad_datatype):
                width = wgt.clad_width
                center_x = (wgt.wg_width + wgt.clad_width) / 2.0
                total_y = sum([layer[1] for layer in mstack.stacklist[key]])
                cur_y = -total_y / 2.0
                for layer in mstack.stacklist[key]:
                    center_y = cur_y + layer[1] / 2.0
                    cur_y = cur_y + layer[1]
                    # Add cladding on +x side
                    CX.append(center_x)
                    CY.append(center_y)
                    width_list.append(width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
                    # Add cladding on -x side
                    CX.append(-center_x)
                    CY.append(center_y)
                    width_list.append(width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])

    elif wgt.wg_type == "slot":
        """Same thing as above but for slot waveguides"""
        slot = wgt.slot
        for key in mstack.stacklist.keys():
            if key == (wgt.wg_layer, wgt.wg_datatype):
                """ Add waveguide blocks """
                rail_width = (wgt.wg_width - slot) / 2.0
                center_x = (slot + rail_width) / 2.0
                total_y = sum([layer[1] for layer in mstack.stacklist[key]])
                cur_y = -total_y / 2.0
                for layer in mstack.stacklist[key]:
                    center_y = cur_y + layer[1] / 2.0
                    cur_y = cur_y + layer[1]
                    # Add left waveguide component
                    CX.append(center_x)
                    CY.append(center_y)
                    width_list.append(rail_width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
                    # Add right waveguide component
                    CX.append(-center_x)
                    CY.append(center_y)
                    width_list.append(rail_width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
            if key == (wgt.clad_layer, wgt.clad_datatype):
                """ Add cladding blocks """
                width = wgt.clad_width
                center_x = (wgt.wg_width + wgt.clad_width) / 2.0
                total_y = sum([layer[1] for layer in mstack.stacklist[key]])
                cur_y = -total_y / 2.0
                for layer in mstack.stacklist[key]:
                    center_y = cur_y + layer[1] / 2.0
                    cur_y = cur_y + layer[1]
                    # Add cladding on +x side
                    CX.append(center_x)
                    CY.append(center_y)
                    width_list.append(width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
                    # Add cladding on -x side
                    CX.append(-center_x)
                    CY.append(center_y)
                    width_list.append(width)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])
                    # Add slot region
                    CX.append(0.0)
                    CY.append(center_y)
                    width_list.append(slot)
                    height_list.append(layer[1])
                    eps_list.append(layer[0])

    if (wgt.wg_width + 2 * wgt.clad_width) < sx:
        """ If True, need to add additional region next to cladding (default material) """
        default_key = (-1, -1)
        center_x = sx / 2.0
        width = sx - (wgt.wg_width + 2 * wgt.clad_width)
        total_y = sum([layer[1] for layer in mstack.stacklist[default_key]])
        cur_y = -total_y / 2.0
        for layer in mstack.stacklist[default_key]:
            center_y = cur_y + layer[1] / 2.0
            cur_y = cur_y + layer[1]
            # Add default material blocks on +x side
            CX.append(center_x)
            CY.append(center_y)
            width_list.append(width)
            height_list.append(layer[1])
            eps_list.append(layer[0])
            # Add default material blocks on -x side
            CX.append(-center_x)
            CY.append(center_y)
            width_list.append(width)
            height_list.append(layer[1])
            eps_list.append(layer[0])

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("CX", data=np.array(CX))
        hf.create_dataset("CY", data=np.array(CY))
        hf.create_dataset("width_list", data=np.array(width_list))
        hf.create_dataset("height_list", data=np.array(height_list))
        hf.create_dataset("eps_list", data=np.array(eps_list))


def _mpb_mode(
    wgt,
    mstack,
    res: int = 10,
    wavelength: float = 1.55,
    sx: float = 4.0,
    sy: float = 4.0,
    plot_mode_number: int = 1,
    polarization: str = "TE",
    epsilon_file: str = "epsilon.h5",
    savefig=False,
    plot: bool = True,
    plotH: bool = False,
    nbackground: float = 1.44,
    dirpath: str = "modes",
):
    """
    Args:
        wgt: waveguide template
        mstack: material stack
        res: Resolution of the simulation [pixels/um] (default=10)
        wavelength: Wavelength in microns (default=1.55)
        sx: Size of the simulation region in the x-direction (default=4.0)
        sy: Size of the simulation region in the y-direction (default=4.0)
        plot_mode_number: Which mode to plot (only plots one mode at a time).  Must be a number equal to or less than num_mode (default=1)
        polarization: If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
        epsilon_file: Filename with the dielectric "block" objects (default=None)
        dirpath: If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
        savefig: Save the mode image to a separate file (default=False)
        plot: plot mode in matplotlib (default=False)

    Returns:
        neff, ng
    """

    eps_input_file = str("epsilon.h5")
    export_wgt_to_hdf5(filename=eps_input_file, wgt=wgt, mstack=mstack, sx=sx)

    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sx))

    with h5py.File(epsilon_file, "r") as hf:
        data = np.array(
            [
                np.array(hf.get("CX")),
                np.array(hf.get("CY")),
                np.array(hf.get("width_list")),
                np.array(hf.get("height_list")),
                np.array(hf.get("eps_list")),
            ]
        )
    geometry = []
    for i in range(len(data[0])):
        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, data[3][i], data[2][i]),
                center=mp.Vector3(0, data[1][i], data[0][i]),
                material=mp.Medium(epsilon=data[4][i]),
            )
        )

    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=res,
        default_material=mp.Medium(epsilon=nbackground ** 2),
        num_bands=plot_mode_number,
    )
    freq = 1 / wavelength
    kdir = mp.Vector3(1, 0, 0)
    tol = 1e-6
    kmag_guess = freq * 2.02
    kmag_min = freq * 0.01
    kmag_max = freq * 10.0

    if polarization == "TE":
        parity = mp.ODD_Z
    elif polarization == "TM":
        parity = mp.EVEN_Z
    elif polarization == "None":
        parity = mp.NO_PARITY

    k = ms.find_k(
        parity,
        freq,
        plot_mode_number,
        plot_mode_number,
        kdir,
        tol,
        kmag_guess,
        kmag_min,
        kmag_max,
    )
    vg = ms.compute_group_velocities()
    k = k[0]
    vg = vg[0][0]

    if plot:
        """
        First plot electric field
        """
        eps = ms.get_epsilon()
        ms.get_dfield(plot_mode_number)
        E = ms.get_efield(plot_mode_number)
        Eabs = np.sqrt(
            np.multiply(E[:, :, 0, 2], E[:, :, 0, 2])
            + np.multiply(E[:, :, 0, 1], E[:, :, 0, 1])
            + np.multiply(E[:, :, 0, 0], E[:, :, 0, 0])
        )
        H = ms.get_hfield(plot_mode_number)
        Habs = np.sqrt(
            np.multiply(H[:, :, 0, 2], H[:, :, 0, 2])
            + np.multiply(H[:, :, 0, 1], H[:, :, 0, 1])
            + np.multiply(H[:, :, 0, 0], H[:, :, 0, 0])
        )

        plt_extent = [-sy / 2.0, +sy / 2.0, -sx / 2.0, +sx / 2.0]

        cmap_fields = "hot_r"
        cmap_geom = "viridis"

        plt.figure(figsize=(14, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(
            abs(E[:, :, 0, 2]),
            cmap=cmap_fields,
            origin="lower",
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|E_x|$")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(
            abs(E[:, :, 0, 1]),
            cmap=cmap_fields,
            origin="lower",
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|E_y|$")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.imshow(
            abs(E[:, :, 0, 0]),
            cmap=cmap_fields,
            origin="lower",
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|E_z|$")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(
            abs(Eabs),
            cmap=cmap_fields,
            origin="lower",
            aspect="auto",
            extent=plt_extent,
        )
        plt.title("Waveguide mode $|E|$")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(
            eps ** 0.5, cmap=cmap_geom, origin="lower", aspect="auto", extent=plt_extent
        )
        plt.title("index profile")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        if savefig:
            dirpath = pathlib.Path(dirpath)
            dirpath.mkdir(exist_ok=True, parents=True)
            plt.savefig(dirpath / f"{polarization}_mode{plot_mode_number}_E.png")

        if plotH:
            """
            Then plot magnetic field
            """
            plt.figure(figsize=(14, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(
                abs(H[:, :, 0, 2]),
                cmap=cmap_fields,
                origin="lower",
                aspect="auto",
                extent=plt_extent,
            )
            plt.title("Waveguide mode $|H_x|$")
            plt.ylabel("y-axis")
            plt.xlabel("x-axis")
            plt.colorbar()

            plt.subplot(2, 3, 2)
            plt.imshow(
                abs(H[:, :, 0, 1]),
                cmap=cmap_fields,
                origin="lower",
                aspect="auto",
                extent=plt_extent,
            )
            plt.title("Waveguide mode $|H_y|$")
            plt.ylabel("y-axis")
            plt.xlabel("x-axis")
            plt.colorbar()

            plt.subplot(2, 3, 3)
            plt.imshow(
                abs(H[:, :, 0, 0]),
                cmap=cmap_fields,
                origin="lower",
                aspect="auto",
                extent=plt_extent,
            )
            plt.title("Waveguide mode $|H_z|$")
            plt.ylabel("y-axis")
            plt.xlabel("x-axis")
            plt.colorbar()

            plt.subplot(2, 3, 4)
            plt.imshow(
                abs(Habs),
                cmap=cmap_fields,
                origin="lower",
                aspect="auto",
                extent=plt_extent,
            )
            plt.title("Waveguide mode $|H|$")
            plt.ylabel("y-axis")
            plt.xlabel("x-axis")
            plt.colorbar()

            plt.subplot(2, 3, 5)
            plt.imshow(
                eps ** 0.5,
                cmap=cmap_geom,
                origin="lower",
                aspect="auto",
                extent=plt_extent,
            )
            plt.title("index profile")
            plt.ylabel("y-axis")
            plt.xlabel("x-axis")
            plt.colorbar()

            plt.tight_layout()
            plt.show()

            if savefig:
                plt.savefig(dirpath / f"{polarization}_mode{plot_mode_number}_H.png")

    neff = wavelength * k
    ng = 1 / vg
    return neff, ng


def mpb_mode(
    ncore=3.55,
    nclad=1.444,
    nbackground: float = 1.44,
    sim_height=4.0,
    wg_width=0.5,
    clad_offset=3.0,
    wg_thickness=0.22,
    slab_thickness=0,
    res: int = 10,
    wavelength: float = 1.55,
    sx: float = 4.0,
    sy: float = 4.0,
    plot_mode_number: int = 1,
    polarization: str = "TE",
    epsilon_file: str = "epsilon.h5",
    savefig=False,
    plot: bool = False,
    plotH: bool = False,
    dirpath: str = "modes",
):
    """

    Args:
        ncore: core refractive index
        nclad: clad refractive index
        nbackground: background refractive index
        sim_height:
        wg_width:
        wg_thickness:
        slab_thickness:
        res: Resolution of the simulation [pixels/um] (default=10)
        wavelength: Wavelength in microns (default=1.55)
        sx: Size of the simulation region in the x-direction (default=4.0)
        sy: Size of the simulation region in the y-direction (default=4.0)
        plot_mode_number: Which mode to plot (only plots one mode at a time).  Must be a number equal to or less than num_mode (default=1)
        polarization: If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
        epsilon_file: Filename with the dielectric "block" objects (default=None)
        dirpath: If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
        savefig: Save the mode image to a separate file (default=False)
        plot: plot mode in matplotlib (default=False)

    Returns:
        neff, ng
    """
    eps_clad = nclad ** 2
    eps_core = ncore ** 2

    box = (sim_height - wg_thickness) / 2
    clad_ridge = sim_height - box - wg_thickness
    clad_slab = sim_height - box - slab_thickness

    etch_stack = [(eps_clad, box), (eps_core, wg_thickness), (eps_clad, clad_ridge)]
    waveguide_stack = [
        (eps_clad, box),
        (eps_core, wg_thickness),
        (eps_clad, clad_ridge),
    ]
    clad_stack = [(eps_clad, box), (eps_core, slab_thickness), (eps_clad, clad_slab)]

    mstack = MaterialStack(
        vsize=sim_height, default_stack=etch_stack, name="Si waveguide"
    )

    mstack.addVStack(layer=LAYER.WG[0], datatype=LAYER.WG[1], stack=waveguide_stack)
    mstack.addVStack(layer=LAYER.WGCLAD[0], datatype=LAYER.WGCLAD[1], stack=clad_stack)

    sx += wg_width
    sy += wg_thickness

    return _mpb_mode(
        wgt=wg_strip(wg_width=wg_width, clad_offset=clad_offset),
        mstack=mstack,
        res=res,
        wavelength=wavelength,
        sx=sx,
        sy=sy,
        plot_mode_number=plot_mode_number,
        polarization=polarization,
        epsilon_file=epsilon_file,
        savefig=savefig,
        plot=plot,
        plotH=plotH,
        dirpath=dirpath,
        nbackground=nbackground,
    )


if __name__ == "__main__":
    neff, ng = mpb_mode(plot=True, wg_width=1.8)
    plt.show()
    print(neff, ng)
