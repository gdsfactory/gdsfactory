# # Monte-Carlo implant simulations
#
# To go beyond implant tables and perform implant simulations on complex geometries, you can call install [PySRIM](https://pypi.org/project/pysrim/) with `pip install pysrim`
#
# Thanks to this plugin you can directly interface with the implant layers of your Components for drift-diffusion simulations and use the results of the ion implantation simulations
#
# [SRIM is a Visual Basic GUI program](http://www.srim.org/) which has been thoroughly benchmarked.
# The website contains lots of documentation on the physics and software.
# It is closed source, but is free to use, copy, modify and distributed for any non-commercial purpose.
# To install it, you can follow the instructions on the [PySRIM repository](https://gitlab.com/costrouc/pysrim/).
# You can install the Windows executable yourself (using Wine on MacOS/Linux), or use a Docker image.
# [The issues contain good information if you run into problems.](https://gitlab.com/costrouc/pysrim/-/issues/7)

from shutil import rmtree
from pathlib import Path
from srim import Ion, Layer, Target

# [The following example follows the tutorial from PySRIM](https://gitlab.com/costrouc/pysrim/-/blob/master/examples/notebooks/Analysis.ipynb), adapted for silicon photonic applications.
#
# ## Simulating n-doping of silicon
#
# ### Setup
#
# Let's compute the implant depth for 100 keV Phosphorus (a typical N-implant) into 220-nm thick SOI:

# +
# Define implant
energy = 2.0e5
implant = Ion("P", energy=1.0e5)

# Define layers of target
nm = 10  # units of SRIM are Angstroms
um = 1e4
soi_thickness = 220 * nm
BOX_thickness = 100 * nm  # instead of 3 * um, ions barely make it to BOX

# 220nm pure silicon
soi = Layer(
    {
        "Si": {
            # (float, int, required): Stoichiometry of element (fraction)
            "stoich": 1.0,
            "E_d": 35.0,  # (float, int, optional): Displacement energy [eV]
            # (float, int, optional): Lattice binding energies [eV]. Used for sputtering calculations.
            "lattice": 0.0,
            # (float, int, optional): Surface binding energies [eV]. Used for sputtering calculations.
            "surface": 3.0,
        },
    },
    density=2.3290,  # density [g/cm^3] of material
    width=soi_thickness,  # width [Angstroms] of layer
)

# 3um SiO2
box = Layer(
    {
        "Si": {
            "stoich": 0.33,
        },
        "O": {
            "stoich": 0.67,
        },
    },
    density=2.65,
    width=BOX_thickness,
)

# Define multilayer target
target = Target([soi, box])
# -

# ### Executing simulation
#
# PySRIM recommends fragmenting the simulation to avoid SRIM crashing:

# +
from gdsfactory.simulation.process.pysrim import run_fragmented_calculation

overwrite = True

srim_executable_directory = Path("/home/bilodeaus/.wine/drive_c/SRIM")
srim_data_directory = Path("./tmp/")

srim_data_directory.mkdir(exist_ok=True, parents=True)
srim_executable_directory.mkdir(exist_ok=True, parents=True)

trim_settings = {
    "calculation": 1,
    "angle_ions": 20,  # exaggerated angle to see its effect
    "ranges": True,
    "plot_mode": 5,
}

if overwrite:
    rmtree(srim_data_directory)
    trim = run_fragmented_calculation(
        srim_executable_directory=srim_executable_directory,
        ion=implant,
        target=target,
        number_ions=10,
        save_path=srim_data_directory,
        trim_settings=trim_settings,
    )
# -

# If you are using your own TRIM installation, you should see a window popup and run the calculations. If using Docker, the process will hang until it is done (there is no progress monitor).
#
# You can also run these in parallel on a multicore machine:

# +
from gdsfactory.simulation.process.pysrim import run_parallel_fragmented_calculation

if overwrite:
    rmtree(srim_data_directory, ignore_errors=True)
    run_parallel_fragmented_calculation(
        srim_executable_directory=srim_executable_directory,
        ion=implant,
        target=target,
        number_ions=10000,
        save_path=srim_data_directory,
        trim_settings=trim_settings,
        step=1000,
        cores=6,
    )
# -

# ### Analyzing vertical distribution
#
# Since we are mainly interested in implant distribution, we can quickly parse the result tree and only extract this information:

# +
from gdsfactory.simulation.process.pysrim import read_ranges

df = read_ranges(srim_data_directory)
# -

# The 'z' direction is depth in the sample, and hence distribution starts at 0:

ax = df.plot.hist(column=["z"], bins=100, alpha=0.5, xlabel="z (A)", density=True)

# The 'y' direction represents lateral scattering in the sample, and hence is centered at 0:

ax = df.plot.hist(column=["y"], bins=100, alpha=0.5, xlabel="x (A)", density=True)

# The x-direction is also lateral, but is along the implantation tilt angle, which results in a skewed distribution for large angles:

ax = df.plot.hist(column=["x"], bins=100, alpha=0.5, xlabel="y (A)", density=True)
