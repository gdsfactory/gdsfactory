""" Convert Components polygons to hdf5 epsilon files for MEEP and MPB  simulations.
Based on PICwriter
"""
import pathlib
import os
import time
from subprocess import call

from matplotlib import pyplot as plt
import numpy as np
import gdspy
import h5py

from pp.layers import LAYER
from pp.sp.meep.mpb_mode import MaterialStack


def export_component_to_hdf5(filename, component, mstack, boolean_operations):
    """Outputs the polygons corresponding to the desired component and MaterialStack.
    Format is compatible for generating prism geometries in MEEP/MPB.

    **Note**: that the top-down view of the device is the 'X-Z' plane.  The 'Y' direction specifies the vertical height.

    Args:
       * **filename** (string): Filename to save (must end with '.h5')
       * **component** (gdspy.Cell): Cell object (component of the PICwriter library)
       * **mstack** (MaterialStack): MaterialStack object that maps the gds layers to a physical stack
       * **boolean_operations** (list): A list of specified boolean operations to be performed on the layers (order matters, see below).

    The boolean_operations argument must be specified in the following format::

       boolean_opeartions = [((layer1/datatype1), (layer2/datatype2), operation), ...]

    where 'operation' can be 'xor', 'or', 'and', or 'not' and the resulting polygons are placed on (layer1, datatype1).
    For example, the boolean_operation below::

       boolean_operations = [((-1,-1), (2,0), 'and'), ((2,0), (1,0), 'xor')]

    will:

       (1) do an 'xor' of the default layerset (-1,-1) with the cladding (2,0) and then make this the new default
       (2) do an 'xor' of the cladding (2,0) and waveguide (1,0) and make this the new cladding

    Write format:
       * LL = layer
       * DD = datatype
       * NN = polygon index
       * VV = vertex index
       * XX = x-position
       * ZZ = z-position
       * height = height of the prism
       * eps = epsilon of the prism
       * y-center = center (y-direction) of the prism [note: (x,y) center defaults to (0,0)]

    """

    flatcell = component.flatten()
    polygon_dict = flatcell.get_polygons(by_spec=True)

    bb = flatcell.get_bounding_box()
    # sx, sy, sz = bb[1][0] - bb[0][0], mstack.vsize, bb[1][1] - bb[0][1]
    center_x, _, center_z = (
        (bb[1][0] + bb[0][0]) / 2.0,
        0.0,
        (bb[1][1] + bb[0][1]) / 2.0,
    )

    ll_list, dd_list, nn_list, vv_list, xx_list, zz_list = [], [], [], [], [], []
    height_list, eps_list, ycenter_list = [], [], []

    """ Add the default layer set """
    polygon_dict[(-1, -1)] = [
        np.array(
            [
                [bb[0][0], bb[0][1]],
                [bb[1][0], bb[0][1]],
                [bb[1][0], bb[1][1]],
                [bb[0][0], bb[1][1]],
            ]
        )
    ]

    #    print("boolean operations: ")
    #    print(boolean_operations)
    #
    #    print('key (default): (-1,-1)')
    #    print(polygon_dict[(-1,-1)])

    for key in polygon_dict.keys():
        """Merge the polygons
        This prevents weird edge effects in MEEP with subpixel averaging between adjacent objects
        """
        polygons = polygon_dict[key]
        polygons_union = gdspy.fast_boolean(
            polygons, polygons, "or", max_points=99999, layer=key[0], datatype=key[1]
        )
        polygon_dict[key] = polygons_union.polygons

    for bo in boolean_operations:
        polygons_bool = gdspy.fast_boolean(
            polygon_dict[bo[0]],
            polygon_dict[bo[1]],
            bo[2],
            layer=bo[0][0],
            datatype=bo[0][1],
        )
        if polygons_bool is None:
            del polygon_dict[bo[0]]
        else:
            polygon_dict[bo[0]] = polygons_bool.polygons

    #    for key in polygon_dict.keys():
    #        print('key: '+str(key))
    #        print(polygon_dict[key])

    for key in polygon_dict.keys():
        ll, dd, = (
            key[0],
            key[1],
        )
        if key in list(mstack.stacklist.keys()):
            stacklist = np.array(mstack.stacklist[key])

            # Put together a list of the centers of each layer
            zlength = sum(stacklist[:, 1])
            z0 = -zlength / 2.0
            centers = [z0 + stacklist[0][1] / 2.0]
            for i in range(len(stacklist) - 1):
                prev_value = centers[-1]
                centers.append(
                    prev_value + stacklist[i][1] / 2.0 + stacklist[i + 1][1] / 2.0
                )

            for i in range(len(stacklist)):
                for nn in range(len(polygon_dict[key])):
                    #                    print("Polygon: ")
                    #                    print("layer=("+str(ll)+"/"+str(dd)+"), height="+str(stacklist[i][1])+" eps="+str(stacklist[i][0])+" ycent="+str(centers[i]))
                    for vv in range(len(polygon_dict[key][nn])):
                        xx, zz = polygon_dict[key][nn][vv]
                        ll_list.append(ll)
                        dd_list.append(dd)
                        nn_list.append(nn)
                        vv_list.append(vv)
                        xx_list.append(xx - center_x)
                        zz_list.append(zz - center_z)
                        height_list.append(stacklist[i][1])
                        eps_list.append(stacklist[i][0])
                        ycenter_list.append(centers[i])

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("LL", data=np.array(ll_list))
        hf.create_dataset("DD", data=np.array(dd_list))
        hf.create_dataset("NN", data=np.array(nn_list))
        hf.create_dataset("VV", data=np.array(vv_list))
        hf.create_dataset("XX", data=np.array(xx_list))
        hf.create_dataset("ZZ", data=np.array(zz_list))
        hf.create_dataset("height", data=np.array(height_list))
        hf.create_dataset("eps", data=np.array(eps_list))
        hf.create_dataset("ycenter", data=np.array(ycenter_list))


def export_timestep_fields_to_png(directory):
    filename = "meept.py"

    """ Export the epsilon slices to images """
    call(
        "h5topng -S3 -m1 -M4 "
        + str(directory)
        + "/topview-"
        + str(filename)
        + "-eps-000000.00.h5",
        shell=True,
    )
    call(
        "h5topng -S3 -m1 -M4 "
        + str(directory)
        + "/sideview-"
        + str(filename)
        + "-eps-000000.00.h5",
        shell=True,
    )

    """ Export the slice of data with epsilon overlayed """
    simulation_time = np.array(
        h5py.File(str(directory) + "/" + str(filename) + "-ez-topview.h5", "r")["ez"]
    ).shape[2]
    simulation_time = simulation_time - 1  # since time starts at t=0

    """ Convert h5 slices to png with dielectric overlayed """
    exec_str = (
        "h5topng -t 0:"
        + str(simulation_time)
        + " -R -Zc dkbluered -a yarg -A "
        + str(directory)
        + "/topview-"
        + str(filename)
        + "-eps-000000.00.h5 "
        + str(directory)
        + "/"
        + str(filename)
        + "-ez-topview.h5"
    )
    call(exec_str, shell=True)
    exec_str = (
        "h5topng -t 0:"
        + str(simulation_time)
        + " -R -Zc dkbluered -a yarg -A "
        + str(directory)
        + "/sideview-"
        + str(filename)
        + "-eps-000000.00.h5 "
        + str(directory)
        + "/"
        + str(filename)
        + "-ez-sideview.h5"
    )
    call(exec_str, shell=True)


def meept(
    component,
    ncore: float = 3.55,
    nclad: float = 1.444,
    sim_height: float = 4.0,
    wg_width: float = 0.5,
    clad_offset: float = 3.0,
    wg_thickness: float = 0.22,
    slab_thickness: float = 0.0,
    port_vcenter=0,
    port_height=0,
    port_width=0,
    res=10,
    wl_center=1.55,
    wl_span=0.3,
    boolean_operations=None,
    input_pol="TE",
    nfreq=100,
    dpml=0.5,
    fields=False,
    source_offset=0.1,
    skip_sim=False,
    dirpath=None,
    parallel=False,
    n_p=2,
):
    """Launches a MEEP simulation to compute the transmission/reflection spectra from each of the component's ports when light enters at the input `port`.

    How this function maps the GDSII layers to the material stack is something that will be improved in the future.  Currently works well for 1 or 2 layer devices.
    **Currently only supports components with port-directions that are `EAST` (0) or `WEST` (pi)**

    Args:
        component: gdsfactory component
        ncore: index core
        nclad: index cladding
        sim_height: simulation region height (um)
        wg_width: width of the waveguide (um)
        clad_offset: offset of the cladding (um)
        wg_thickness:  (um)
        slab_thickness:  (um)
        ports (list of `Port` dicts): These are the ports to track the Poynting flux through.  **IMPORTANT** The first element of this list is where the Eigenmode source will be input.
        port_vcenter (float): Vertical center of the waveguide
        port_height (float): Height of the port cross-section (flux plane)
        port_width (float): Width of the port cross-section (flux plane)
        res (int): Resolution of the MEEP simulation
        wl_center (float): Center wavelength (in microns)
        wl_span (float): Wavelength span (determines the pulse width)
        boolean_operations (list): A list of specified boolean operations to be performed on the layers (ORDER MATTERS).  In the following format:
        [((layer1/datatype1), (layer2/datatype2), operation), ...] where 'operation' can be 'xor', 'or', 'and', or 'not' and the resulting polygons are placed on (layer1, datatype1).  See below for example.
        input_pol (String): Input polarization of the waveguide mode.  Must be either "TE" or "TM".  Defaults to "TE" (z-antisymmetric).
        nfreq (int): Number of frequencies (wavelengths) to compute the spectrum over.  Defaults to 100.
        dpml (float): Length (in microns) of the perfectly-matched layer (PML) at simulation boundaries.  Defaults to 0.5 um.
        fields (boolean): If true, outputs the epsilon and cross-sectional fields.  Defaults to False.
        source_offset (float): Offset (in x-direction) between reflection monitor and source.  Defaults to 0.1 um.
        skip_sim (boolean): Defaults to False.  If True, skips the simulation (and hdf5 export).  Useful if you forgot to perform a normalization and don't want to redo the whole MEEP simulation.
        dirpath (string): Output directory for files generated.  Defaults to 'meep-componentName'.
        parallel (boolean): If `True`, will run simulation on `np` cores (`np` must be specified below, and MEEP/MPB must be built from source with parallel-libraries).  Defaults to False.
        n_p (int): Number of processors to run meep simulation on.  Defaults to `2`.


    Example of **boolean_operations** (using the default):
           The following default boolean_operation will:
               (1) do an 'xor' of the default layerset (-1,-1) with a cladding (2,0) and then make this the new default layerset
               (2) do an 'xor' of the cladding (2,0) and waveguide (1,0) and make this the new cladding

            boolean_operations = [((-1,-1), (2,0), 'and'), ((2,0), (1,0), 'xor')]

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

    ports = component.get_ports_list()

    dirpath = dirpath or f"meep_{component.name}"
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(exist_ok=True, parents=True)

    boolean_operations = []
    # wgt = wg_strip(wg_width=wg_width, clad_offset=clad_offset)

    # if boolean_operations is None:
    #     boolean_operations = [
    #         ((-1, -1), (wgt.clad_layer, wgt.clad_datatype), "xor"),
    #         (
    #             (wgt.clad_layer, wgt.clad_datatype),
    #             (wgt.wg_layer, wgt.wg_datatype),
    #             "xor",
    #         ),
    #     ]

    """ For each port determine input_direction (useful for computing the sign of the power flux) """
    input_directions = []
    for port in ports:
        orientation = port.orientation
        if orientation == 0:
            input_directions.append(-1)
        elif orientation == 180:
            input_directions.append(1)
        else:
            raise ValueError(
                f"port {port.name} orientation = {orientation}, only 0 or 180 degrees supported"
            )

    # Get size, center of simulation window
    flatcell = component.flatten()
    bb = flatcell.get_bounding_box()
    sx, sy, sz = bb[1][0] - bb[0][0], mstack.vsize, bb[1][1] - bb[0][1]
    center = ((bb[1][0] + bb[0][0]) / 2.0, 0, (bb[1][1] + bb[0][1]) / 2.0)

    # Convert the structure to an hdf5 file
    eps_input_file = dirpath / f"{component.name}.h5"
    logpath = dirpath / f"{component.name}.log"
    datapath = dirpath / f"{component.name}.dat"
    export_component_to_hdf5(eps_input_file, component, mstack, boolean_operations)

    # Launch MEEP simulation using correct inputs
    port_string = ""
    for port in ports:
        x, y = port.midpoint
        port_string += f"{x} {y} "
    port_string = str(port_string[:-1])

    exec_str = f"python meept.py -fields {fields} -input_pol {input_pol} -output_directory {dirpath} -eps_input_file {eps_input_file} -res {res} -nfreq {nfreq} -input_direction {input_directions[0]} -dpml {dpml} -wl_center {wl_center:.3f} "
    exec_str += f"-wl_span {wl_span:.3f} -port_vcenter {port_vcenter:.3f} -port_height {port_height:.3f} -port_width {port_width:.3f} -source_offset {source_offset:.3f} -center_x {center[0]} -center_y {center[1]} -center_z {center[2]} "
    exec_str += f"-sx {sx} -sy {sy} -sz {sz} -port_coords {port_string} > {logpath}"
    if parallel:
        exec_str = f"mpirun -np {int(n_p)} {exec_str}"

    if not skip_sim:
        print(f"Running MEEP simulation... (check {logpath} file for current status)")
        print(exec_str)

        start = time.time()
        call(exec_str, shell=True, cwd=dirpath)
        print(f"Time to run MEEP simulation = {time.time() - start} seconds")

    call(f"grep flux1: {logpath} > {datapath}", shell="True")


def plot_results(
    component,
    wl_center=1.55,
    wl_span=0.3,
    res=10,
    dirpath=None,
    fields=False,
    plot_window=False,
):
    """Grab data and plot transmission/reflection spectra

    plot_window (boolean): If true, outputs the spectrum plot in a matplotlib window (in addition to saving).  Defaults to False.

    """
    dirpath = dirpath or f"meep_{component.name}"
    dirpath = pathlib.Path(dirpath)
    datapath = f"{component.name}.dat"
    comp_data = np.genfromtxt(datapath, delimiter=",")

    ports = component.get_ports_list()
    input_directions = []
    for port in ports:
        orientation = port.orientation
        if orientation == 0:
            input_directions.append(-1)
        elif orientation == 180:
            input_directions.append(1)
        else:
            raise ValueError(
                f"port {port.name} orientation = {orientation}, only 0 or 180 degrees supported"
            )

    # Get the power flux-data from the component simulation for each flux-plane
    flux_data = []
    for i in range(len(ports)):
        flux_data.append((-1) * input_directions[i] * comp_data[:, i + 2])

    # wavelength = [1.0 / f for f in freq]
    wavelength = [wl_center - wl_span, wl_center, wl_center + wl_span]

    # Plot a spectrum corresponding to each port (sign is calculated from the port "direction")
    colorlist = ["r-", "b-", "g-", "c-", "m-", "y-"]
    plt.plot(wavelength, (flux_data[0]), colorlist[0], label="port 0")
    for i in range(len(flux_data) - 1):
        plt.plot(
            wavelength,
            flux_data[i + 1],
            colorlist[(i + 1) % len(colorlist)],
            label="port " + str(i + 1),
        )

    plt.xlabel("Wavelength [um]")
    plt.ylabel("Transmission")
    plt.xlim([min(wavelength), max(wavelength)])
    plt.legend(loc="best")
    plt.savefig("%s/%s-res%d.png" % (str(os.getcwd()), str(dirpath), res))
    if plot_window:
        plt.show()
    plt.close()

    if fields:
        print("Outputting fields images to " + str(dirpath))
        export_timestep_fields_to_png(str(dirpath))


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    meept(c)
