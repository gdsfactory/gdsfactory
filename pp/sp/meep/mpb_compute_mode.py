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


def mpb_compute_mode(
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
    dirpath: str = "modes",
):
    """
    Args:
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

    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)
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
        default_material=mp.Medium(epsilon=1.0),
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
    print("k=" + str(k))
    print("v_g=" + str(vg))

    k = k[0]
    vg = vg[0][0]

    """ Plot modes """
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

    if plot:
        """
        First plot electric field
        """
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
            eps, cmap=cmap_geom, origin="lower", aspect="auto", extent=plt_extent
        )
        plt.title("Waveguide dielectric")
        plt.ylabel("y-axis")
        plt.xlabel("x-axis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        if savefig:
            plt.savefig(dirpath / f"{polarization}_mode{plot_mode_number}_E.png")

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
            eps, cmap=cmap_geom, origin="lower", aspect="auto", extent=plt_extent
        )
        plt.title("Waveguide dielectric")
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


if __name__ == "__main__":
    import pp.sp.meep as ms
    from pp.components.waveguide_template import wg_strip

    epsSiO2 = 1.444 ** 2
    epsSi = 3.55 ** 2
    etch_stack = [(epsSiO2, 1.89), (epsSi, 0.07), (epsSiO2, 2.04)]
    mstack = ms.MaterialStack(vsize=4.0, default_stack=etch_stack, name="Si waveguide")
    waveguide_stack = [(epsSiO2, 1.89), (epsSi, 0.22), (epsSiO2, 1.89)]
    clad_stack = [(epsSiO2, 1.89), (epsSi, 0.05), (epsSiO2, 2.06)]
    mstack.addVStack(layer=1, datatype=0, stack=waveguide_stack)
    mstack.addVStack(layer=2, datatype=0, stack=clad_stack)

    mpb_compute_mode(
        wgt=wg_strip(),
        mstack=mstack,
        res=128,
        wavelength=1.55,
        sx=3.0,
        sy=3.0,
        plot_mode_number=1,
        polarization="TE",
    )

    plt.show()
