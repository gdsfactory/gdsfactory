"""
MCTS = "Meep Compute Transmission Spectra"
Default MEEP launch-file for arbitrary PICwriter components.

Launches a MEEP simulation to compute the transmission/reflection spectra from each of the component's ports when light
enters at the input `port`.

How this function maps the GDSII layers to the material stack is something that will be improved in the future.
Currently works well for 1 or 2 layer devices.

@author: dkita
"""

import numpy as np
import meep as mp
import h5py
import argparse
import operator


def str2bool(v):
    """ Allow proper argparse handling of boolean inputs """
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_prism_objects(eps_file):
    with h5py.File(eps_file, "r") as hf:
        """ Write-format:
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
        data = np.array(
            [
                np.array(hf.get("LL")),
                np.array(hf.get("DD")),
                np.array(hf.get("NN")),
                np.array(hf.get("VV")),
                np.array(hf.get("XX")),
                np.array(hf.get("ZZ")),
                np.array(hf.get("height")),
                np.array(hf.get("eps")),
                np.array(hf.get("ycenter")),
            ]
        )

    """ Now, restructure so that each element of the prisms list contains all the info needed to instantiate:
        i.e. [{'center':(0,0,0), 'height':0.4, 'eps':2.3, 'vlist':[(0,0), (1,2), ...]}]
    """
    prisms = []

    # get all unique (LL,DD,NN,ycenter,height,eps) combinations (each corresponds to a unique prism)
    unique_vals = set([tuple(data[(0, 1, 2, 6, 7, 8), i]) for i in range(len(data[0]))])
    for val in unique_vals:
        # Search 'data' for all vertex values with the matching vals
        vertex_list = []
        for i in range(len(data[0])):
            if (
                data[0, i] == val[0]
                and data[1, i] == val[1]
                and data[2, i] == val[2]
                and data[6, i] == val[3]
                and data[7, i] == val[4]
                and data[8, i] == val[5]
            ):
                vertex_list.append([data[3, i], data[4, i], data[5, i]])
        # Sort vertices
        vertex_list.sort(key=operator.itemgetter(0))

        # Get rid of the numbering for each vertex
        vl = [
            mp.Vector3(
                float(vertex_list[i][1]),
                float(val[5]) - float(val[3]) / 2.0,
                float(vertex_list[i][2]),
            )
            for i in range(len(vertex_list))
        ]

        prism = {"height": float(val[3]), "eps": float(val[4]), "vlist": vl}
        prisms.append(prism)

        # Sort prisms in order of ascending dielectric constant
        prisms_sorted = sorted(prisms, key=operator.itemgetter("eps"))

    if len(unique_vals) == 0:
        raise ValueError(
            "Epsilon file (eps_file="
            + str(eps_file)
            + ") is empty or the wrong format."
        )

    return prisms_sorted


def main(args):
    """
    Args:
       * **fields** (boolean): If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
       * **output_directory** (string): Name of the output directory (for storing the fields)
       * **eps_input_file** (string): Name of the hdf5 file that defines the geometry through prisms
       * **input_pol** (string): Either "TE", or "TM", corresponding to the desired input mode.  Defaults to "TE"
       * **res** (int): Resolution of the MEEP simulation
       * **nfreq** (int): The number of wavelength points to record in the transmission/reflection spectra
       * **input_direction** (1 or -1): Direction of propagation for the input eigenmode.  If +1, goes in +x, else if -1, goes in -x.  Defaults to +1.
       * **dpml** (float): Length (in microns) of the perfectly-matched layer (PML) at simulation boundaries.  Defaults to 0.5 um.
       * **wl_center** (float): Center wavelength (in microns)
       * **wl_span** (float): Wavelength span (determines the pulse width)
       * **port_vcenter** (float): Vertical center of the waveguide
       * **port_height** (float): Height of the port cross-section (flux plane)
       * **port_width** (float): Width of the port cross-section (flux plane)
       * **source_offset** (float): Offset (in x-direction) between reflection monitor and source.  Defaults to 0.1 um.
       * **center_x** (float): x-coordinate of the center of the simulation region
       * **center_y** (float): y-coordinate of the center of the simulation region
       * **center_z** (float): z-coordinate of the center of the simulation region
       * **sx** (float): Size of the simulation region in x-direction
       * **sx** (float): Size of the simulation region in y-direction
       * **sz** (float): Size of the simulation region in z-direction
       * **port_coords** (list): List of the port coordinates (variable length), in the format [x1, y1, x2, y2, x3, y3, ...] (*must* be even)
    """
    # Boolean inputs
    fields = args.fields

    # String inputs
    output_directory = args.output_directory
    eps_input_file = args.eps_input_file
    input_pol = args.input_pol

    # Int inputs
    res = args.res
    nfreq = args.nfreq
    input_direction = args.input_direction

    # Float inputs
    dpml = args.dpml
    wl_center = args.wl_center
    wl_span = args.wl_span
    port_vcenter = args.port_vcenter
    port_height = args.port_height
    port_width = args.port_width
    source_offset = args.source_offset
    center_x, center_y, center_z = args.center_x, args.center_y, args.center_z
    sx, sy, sz = args.sx, args.sy, args.sz

    # List of floats
    port_coords = [float(x) for x in args.port_coords[0].split(" ")]
    ports = [
        (port_coords[2 * i], port_coords[2 * i + 1])
        for i in range(int(len(port_coords) / 2))
    ]

    if input_pol == "TE":
        parity = mp.ODD_Z
    elif input_pol == "TM":
        parity = mp.EVEN_Z
    else:
        raise ValueError(
            "Warning! Improper value of 'input_pol' was passed to mcts.py (input_pol given ="
            + str(input_pol)
            + ")"
        )

    if len(port_coords) % 2 != 0:
        raise ValueError(
            "Warning! Improper port_coords was passed to `meep_compute_transmission_spectra`.  Must be even number of port_coords in [x1, y1, x2, y2, ..] format."
        )

    # Setup the simulation geometries

    prism_objects = get_prism_objects(eps_input_file)
    geometry = []
    for p in prism_objects:
        #        print('vertices='+str(p['vlist']))
        #        print('axis = '+str(mp.Vector3(0,1,0)))
        #        print('height = '+str(p['height']))
        print("material = " + str(p["eps"]))
        #        print('\n')
        geometry.append(
            mp.Prism(
                p["vlist"],
                axis=mp.Vector3(0, 1, 0),
                height=p["height"],
                material=mp.Medium(epsilon=p["eps"]),
            )
        )

    # Setup the simulation sources
    fmax = 1.0 / (wl_center - 0.5 * wl_span)
    fmin = 1.0 / (wl_center + 0.5 * wl_span)
    fcen = (fmax + fmin) / 2.0
    df = fmax - fmin
    if abs(abs(input_direction) - 1) > 1e-6:
        print(input_direction)
        raise ValueError("Warning! input_direction is not +1 or -1.")

    # Use first port in 'ports' as the location of the eigenmode source
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df, cutoff=30),
            component=mp.ALL_COMPONENTS,
            size=mp.Vector3(0, 3 * float(port_height), 3 * float(port_width)),
            center=mp.Vector3(
                ports[0][0] + source_offset - center_x,
                float(port_vcenter) - center_y,
                ports[0][1] - center_z,
            ),
            eig_match_freq=True,
            eig_parity=parity,
            eig_kpoint=mp.Vector3(float(input_direction) * wl_center, 0, 0),
            eig_resolution=2 * res if res > 16 else 32,
        )
    ]

    # Setup the simulation
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        dimensions=3,
        resolution=res,
        filename_prefix=False,
    )

    """ Add power flux monitors """
    print("ADDING FLUX MONITORS")
    flux_plane_objects = []

    for port in ports:
        flux_region = mp.FluxRegion(
            size=mp.Vector3(0, float(port_height), float(port_width)),
            center=mp.Vector3(
                float(port[0]) - center_x,
                float(port_vcenter) - center_y,
                float(port[1]) - center_z,
            ),
        )
        fpo = sim.add_flux(fcen, df, nfreq, flux_region)
        flux_plane_objects.append(fpo)

    sim.use_output_directory(str(output_directory))

    """ Run the simulation """

    """ Monitor the amplitude in the center of the structure """
    decay_pt = mp.Vector3(0, port_vcenter, 0)

    sv = mp.Volume(size=mp.Vector3(sx, sy, 0), center=mp.Vector3(0, 0, 0))
    tv = mp.Volume(size=mp.Vector3(sx, 0, sz), center=mp.Vector3(0, port_vcenter, 0))

    print("RUNNING SIMULATION")

    if fields:
        sim.run(
            mp.at_beginning(mp.output_epsilon),
            mp.at_beginning(
                mp.with_prefix(str("sideview-"), mp.in_volume(sv, mp.output_epsilon))
            ),
            mp.at_beginning(
                mp.with_prefix(str("topview-"), mp.in_volume(tv, mp.output_epsilon))
            ),
            mp.at_every(
                1.0,
                mp.to_appended(
                    str("ez-sideview"), mp.in_volume(sv, mp.output_efield_z)
                ),
            ),
            mp.at_every(
                1.0,
                mp.to_appended(str("ez-topview"), mp.in_volume(tv, mp.output_efield_z)),
            ),
            until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, decay_pt, 1e-4),
        )
    else:
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, decay_pt, 1e-4)
        )

    sim.display_fluxes(*flux_plane_objects)

    print("FINISHED SIMULATION")


if __name__ == "__main__":
    """
    Args:
       * **fields** (boolean): If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view)
       * **output_directory** (string): Name of the output directory (for storing the fields)
       * **eps_input_file** (string): Name of the hdf5 file that defines the geometry through prisms
       * **res** (int): Resolution of the MEEP simulation
       * **nfreq** (int): The number of wavelength points to record in the transmission/reflection spectra
       * **input_direction** (1 or -1): Direction of propagation for the input eigenmode.  If +1, goes in +x, else if -1, goes in -x.  Defaults to +1.
       * **dpml** (float): Length (in microns) of the perfectly-matched layer (PML) at simulation boundaries.  Defaults to 0.5 um.
       * **wl_center** (float): Center wavelength (in microns)
       * **wl_span** (float): Wavelength span (determines the pulse width)
       * **port_vcenter** (float): Vertical center of the waveguide
       * **port_height** (float): Height of the port cross-section (flux plane)
       * **port_width** (float): Width of the port cross-section (flux plane)
       * **source_offset** (float): Offset (in x-direction) between reflection monitor and source.  Defaults to 0.1 um.
       * **center_x** (float): x-coordinate of the center of the simulation region
       * **center_y** (float): y-coordinate of the center of the simulation region
       * **center_z** (float): z-coordinate of the center of the simulation region
       * **sx** (float): Size of the simulation region in x-direction
       * **sx** (float): Size of the simulation region in y-direction
       * **sz** (float): Size of the simulation region in z-direction
       * **port_coords** (list): List of the port coordinates (variable length), in the format [x1, y1, x2, y2, x3, y3, ...] (*must* be even)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fields",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, outputs the fields at the relevant waveguide cross-sections (top-down and side-view) (default=False)",
    )
    parser.add_argument(
        "-output_directory",
        type=str,
        default=None,
        help="Name of the output directory (for storing the fields) (default=None)",
    )
    parser.add_argument(
        "-eps_input_file",
        type=str,
        default=None,
        help="Name of the hdf5 file that defines the geometry through prisms (default=None)",
    )
    parser.add_argument(
        "-input_pol",
        type=str,
        default="TE",
        help='Either "TE", or "TM", corresponding to the desired input mode polarization (default="TE")',
    )
    parser.add_argument(
        "-res",
        type=int,
        default=10,
        help="Resolution of the simulation [pixels/um] (default=10)",
    )
    parser.add_argument(
        "-nfreq",
        type=int,
        default=100,
        help="Number of frequencies sampled (for flux) between fcen-df/2 and fcen+df/2 (default=100)",
    )
    parser.add_argument(
        "-input_direction",
        type=int,
        default=1,
        help="Direction of propagation for the input eigenmode.  If +1, goes in +x, else if -1, goes in -x (default=+1)",
    )
    parser.add_argument(
        "-dpml",
        type=float,
        default=0.5,
        help="Thickness of the PML region (default=0.5)",
    )
    parser.add_argument(
        "-wl_center",
        type=float,
        default=1.55,
        help="Center wavelength [in um] of the Gaussian pulse (default=1.55)",
    )
    parser.add_argument(
        "-wl_span",
        type=float,
        default=0.300,
        help="Spectral width of the Gaussian pulse [in um] (default=0.300)",
    )
    parser.add_argument(
        "-port_vcenter",
        type=float,
        default=0,
        help="Vertical center of the waveguide [in um] (default=0.0)",
    )
    parser.add_argument(
        "-port_height",
        type=float,
        default=0,
        help="Height of the port cross-section (flux plane) [in um] (default=0.0)",
    )
    parser.add_argument(
        "-port_width",
        type=float,
        default=0,
        help="Width of the port cross-section (flux plane) [in um] (default=0.0)",
    )
    parser.add_argument(
        "-source_offset",
        type=float,
        default=0.1,
        help="Offset (in x-direction) between reflection monitor and source [in um] (default=0.1)",
    )

    parser.add_argument(
        "-center_x",
        type=float,
        default=0,
        help="x-coordinate of the center of the simulation region (default=0.0)",
    )
    parser.add_argument(
        "-center_y",
        type=float,
        default=0,
        help="y-coordinate of the center of the simulation region (default=0.0)",
    )
    parser.add_argument(
        "-center_z",
        type=float,
        default=0,
        help="z-coordinate of the center of the simulation region (default=0.0)",
    )
    parser.add_argument(
        "-sx",
        type=float,
        default=1.0,
        help="Size of the simulation region in x-direction (default=1.0)",
    )
    parser.add_argument(
        "-sy",
        type=float,
        default=1.0,
        help="Size of the simulation region in y-direction (default=1.0)",
    )
    parser.add_argument(
        "-sz",
        type=float,
        default=1.0,
        help="Size of the simulation region in z-direction (default=1.0)",
    )

    parser.add_argument(
        "-port_coords",
        nargs="+",
        required=True,
        help="List of the port coordinates (variable length), in the format [x1, y1, x2, y2, x3, y3, ...] (*must* be even)",
    )

    args = parser.parse_args()
    main(args)
