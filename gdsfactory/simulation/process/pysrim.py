from itertools import count
from pathlib import Path
from srim import TRIM
import os


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
        trim.run(srim_executable_directory)
        save_directory = find_folder(save_path)
        os.makedirs(save_directory, exist_ok=True)
        TRIM.copy_output_files(srim_executable_directory, save_directory)
