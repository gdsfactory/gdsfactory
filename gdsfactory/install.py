"""Install Klayout and GIT plugins."""
from __future__ import annotations

import configparser
import os
import pathlib
import shutil
import sys
from typing import Optional
from gdsfactory.config import PATH

home = pathlib.Path.home()


def remove_path_or_dir(dest: pathlib.Path) -> None:
    if dest.is_symlink():
        os.unlink(dest)
    elif dest.is_dir():
        shutil.rmtree(dest)
    else:
        os.remove(dest)


def make_link(src, dest, overwrite: bool = True) -> None:
    dest = pathlib.Path(dest)
    if dest.exists() and not overwrite:
        print(f"{dest} already exists")
        return
    if dest.exists() or dest.is_symlink():
        print(f"removing {dest} already installed")
        remove_path_or_dir(dest)
    try:
        os.symlink(src, dest, target_is_directory=True)
    except OSError as err:
        print("Could not create symlink!")
        print("     Error: ", err)
        if sys.platform == "win32":
            # https://stackoverflow.com/questions/32877260/privlege-error-trying-to-create-symlink-using-python-on-windows-10
            shutil.copy(src, dest)
    print("Symlink made:")
    print(f"From: {src}")
    print(f"To:   {dest}")


def install_gdsdiff() -> None:
    """Install gdsdiff tool for GIT."""
    print("gdsdiff shows boolean differences in Klayout so you can run:")
    print("git diff FILE.GDS")
    print("Adding gdsdiff command to ~/.gitconfig and ~/.config/git/attributes")
    _write_git_config()
    _write_git_attributes()


def _write_git_config() -> None:
    """Write GIT config in ~/.gitconfig."""
    git_config_path = home / ".gitconfig"
    config = configparser.RawConfigParser()
    config.read(git_config_path)
    key = 'diff "gdsdiff"'

    if key not in config.sections():
        config.add_section(key)
        config.set(key, "command", "python -m gdsfactory.difftest_git")
        config.set(key, "binary", "True")

        with open(git_config_path, "w+") as f:
            config.write(f, space_around_delimiters=True)


def _write_git_attributes() -> None:
    """Write git attributes in ~/.config/git/attributes."""
    git_config = home / ".config" / "git"
    git_config.mkdir(exist_ok=True, parents=True)
    line_to_add = "*.gds diff=gdsdiff\n"

    # Specify the path to the .config/git/attributes file
    dirpath = home / ".config" / "git"
    dirpath.mkdir(exist_ok=True, parents=True)
    file_path = dirpath / "attributes"

    # Read the file to check if the line already exists
    file_content = file_path.read_text() if file_path.exists() else ""

    # Add the line only if it doesn't exist
    if line_to_add not in file_content:
        with open(file_path, "a") as file:
            file.write(line_to_add)


def get_klayout_path() -> pathlib.Path:
    """Returns KLayout path."""
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    return home / klayout_folder


def copy(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy overwriting file or directory."""
    dest_folder = dest.parent
    dest_folder.mkdir(exist_ok=True, parents=True)

    if dest.exists() or dest.is_symlink():
        print(f"removing {dest} already installed")
        remove_path_or_dir(dest)

    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy(src, dest)
    print(f"{src} copied to {dest}")


def _install_to_klayout(
    src: pathlib.Path, klayout_subdir_name: str, package_name: str
) -> None:
    """Install into KLayout technology.

    Equivalent to using KLayout package manager.

    """
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    subdir = home / klayout_folder / klayout_subdir_name
    dest = subdir / package_name
    subdir.mkdir(exist_ok=True, parents=True)
    make_link(src, dest)


def install_klayout_package() -> None:
    """Install gdsfactory KLayout package.

    Equivalent to using KLayout package manager.
    """
    cwd = pathlib.Path(__file__).resolve().parent
    _install_to_klayout(
        src=cwd / "generic_tech" / "klayout",
        klayout_subdir_name="salt",
        package_name="gdsfactory",
    )


def install_klayout_technology(
    tech_dir: pathlib.Path, tech_name: Optional[str] = None
) -> None:
    """Install technology to KLayout."""
    _install_to_klayout(
        src=tech_dir,
        klayout_subdir_name="tech",
        package_name=tech_name or tech_dir.name,
    )


py_files = PATH.notebooks.glob("**/*.py")


def convert_py_to_ipynb(files=py_files, output_folder=PATH.cwd / "notebooks") -> None:
    """Convert notebooks from markdown to ipynb."""
    import jupytext

    output_folder.mkdir(exist_ok=True, parents=True)

    for file in files:
        notebook_file = f"{output_folder}/{file.stem}.ipynb"
        nb = jupytext.read(file)
        jupytext.write(nb, notebook_file)


if __name__ == "__main__":
    # cwd = pathlib.Path(__file__).resolve().parent
    # home = pathlib.Path.home()
    # src = cwd / "generic_tech" / "klayout" / "tech"

    # write_git_attributes()
    # install_gdsdiff()
    # install_klayout_package()
    convert_py_to_ipynb()
