from itertools import count
from pathlib import Path
import shutil
from srim import TRIM
import os
from itertools import repeat
import pandas as pd


def fragment(step, total):
    remaining = total
    while remaining > 0:
        if step > remaining:
            return remaining
        else:
            remaining -= step
            yield step


def find_folder(directory):
    for i in count():
        path = Path(directory) / str(i)
        if not path.is_dir():
            return str(path.absolute())


def run_fragmented_calculation(
    srim_executable_directory: Path,
    ion,
    target,
    number_ions,
    save_path: Path,
    trim_settings=None,
    step: int = 1000,
):
    """Runs a TRIM calculations in series, with each batch no more than 1000 ions to avoid crashes.

    Arguments:
        srim_executable_directory: directory where running "wine TRIM" opens the software
        ion: pysrim ion
        target: pysrim target
        number_ions: number of ions to simulate
        path: where to save to save data
        trim_settings: dict of SRIM simulation settings
        step: number of simulations per batch. Default 1000.
    """
    for i, num_ions in enumerate(fragment(step, number_ions)):
        print(
            "total ions completed: {:06d}\tion: {}\tions in step: {:06d}".format(
                i * step, ion.symbol, num_ions
            )
        )
        trim_settings = trim_settings or {"calculation": 2}
        trim = TRIM(target, ion, number_ions=num_ions, **trim_settings)
        print(srim_executable_directory)
        trim.run(srim_executable_directory)
        save_directory = find_folder(save_path)
        os.makedirs(save_directory, exist_ok=True)
        TRIM.copy_output_files(srim_executable_directory, save_directory)


def run_parallel_fragmented_calculation(
    srim_executable_directory: Path,
    ion,
    target,
    number_ions,
    save_path: Path,
    trim_settings=None,
    step: int = 1000,
    cores: int = 4,
):
    """Runs "cores" TRIM series calculations in parallel, with each batch no more than 1000 ions to avoid crashes.

    Arguments:
        cores: number of parallel SRIM instances to run.
    """
    from multiprocessing import Pool

    number_ions_per_core = int(number_ions / cores)

    # First deduplicate the TRIM executable to avoid results conflicts
    for i in range(cores):
        if Path(srim_executable_directory.parent / str(i)).exists():
            shutil.rmtree(srim_executable_directory.parent / str(i))
        shutil.copytree(
            srim_executable_directory, srim_executable_directory.parent / str(i)
        )

    with Pool(cores) as p:
        p.starmap(
            run_fragmented_calculation,
            zip(
                [srim_executable_directory.parent / str(i) for i in range(cores)],
                repeat(ion),
                repeat(target),
                repeat(number_ions_per_core),
                [save_path / str(i) for i in range(cores)],
                repeat(trim_settings),
                repeat(step),
            ),
        )

    # Delete redundant executables
    for i in range(cores):
        shutil.rmtree(srim_executable_directory.parent / str(i))


def read_ranges(save_path):
    """Read ranges from a data directory (or subdirectories).

    TODO: contribute back to PySRIM.

    Returns:
        Dataframe with all ion final positions.
        z-column represents depth in the sample
        x-column represent lateral dimension, in tilt angle direction
        y-column represents lateral dimension, perpendicular to tilt angle direction
    """
    dataframes = []
    for root, _dirs, files in os.walk(save_path):
        for filename in files:
            if filename == "RANGE_3D.txt":
                with open(Path(root) / filename, "r", encoding="latin-1") as f:
                    # Find table
                    dataframes.extend(
                        pd.read_csv(f, sep="\s+", header=None)[[1, 2, 3]]  # noqa: W605
                        for num, line in enumerate(f, 1)
                        if "----" in line
                    )
    return pd.concat(dataframes).rename(columns={1: "z", 2: "x", 3: "y"}).reset_index()


if __name__ == "__main__":
    from pathlib import Path
    from srim import Ion, Layer, Target
    import matplotlib.pyplot as plt

    # Define implant
    energy = 1.0e5
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

    # Simulation parameters
    overwrite = True

    srim_executable_directory = Path("/home/bilodeaus/.wine/drive_c/SRIM")
    srim_data_directory = Path("./tmp/")

    srim_data_directory.mkdir(exist_ok=True, parents=True)
    srim_executable_directory.mkdir(exist_ok=True, parents=True)

    trim_settings = {
        "calculation": 1,
        "angle_ions": 45,
        "ranges": True,
        "plot_mode": 5,
    }

    if overwrite:
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

    df = read_ranges(srim_data_directory)

    ax = df.plot.hist(column=["z"], bins=200, alpha=0.5, xlabel="z (A)", density=True)
    ax = df.plot.hist(column=["z"], bins=100, alpha=0.5, xlabel="z (A)", density=True)
    ax = df.plot.hist(column=["x"], bins=100, alpha=0.5, xlabel="x (A)", density=True)
    ax = df.plot.hist(column=["y"], bins=100, alpha=0.5, xlabel="y (A)", density=True)
    plt.show()
