"""Install Klayout and GIT plugins."""
import configparser
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Optional


def remove_path_or_dir(dest: pathlib.Path):
    if dest.is_dir():
        if dest.is_symlink():
            os.rmdir(dest)
        else:
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
            print("Trying to create a junction instead of a symlink...")
            proc = subprocess.check_call(f"mklink /J {dest} {src}", shell=True)
            if proc != 0:
                print("Could not create link!")
    print("Symlink made:")
    print(f"From: {src}")
    print(f"To:   {dest}")


def install_gdsdiff() -> None:
    """Install gdsdiff tool."""
    home = pathlib.Path.home()
    git_config_path = home / ".gitconfig"
    git_attributes_path = home / ".gitattributes"

    if git_config_path.exists():
        git_config_str = open(git_config_path).read()
    else:
        git_config_str = "empty"

    git_attributes_str = (
        open(git_attributes_path).read() if git_attributes_path.exists() else "empty"
    )

    if "gds_diff" not in git_config_str:
        write_git_config(git_config_path)
    if "gds_diff" not in git_attributes_str:
        print("Appending the gdsdiff command to your ~/.gitattributes")

        with open(git_attributes_path, "a") as f:
            f.write("*.gds diff=gds_diff\n")


def write_git_config(git_config_path):
    """Write GIT config."""
    print("gdsdiff shows boolean differences in Klayout")
    print("git diff FILE.GDS")
    print("Appending the gdsdiff command to your ~/.gitconfig")

    config = configparser.RawConfigParser()
    config.read(git_config_path)
    key = 'diff "gds_diff"'

    if key not in config.sections():
        config.add_section(key)
        config.set(key, "command", "python -m gdsfactory.gdsdiff.gds_diff_git")
        config.set(key, "binary", "True")

        with open(git_config_path, "w+") as f:
            config.write(f, space_around_delimiters=True)


def get_klayout_path() -> pathlib.Path:
    """Returns klayout path."""
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    home = pathlib.Path.home()
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

    Equivalent to using klayout package manager.

    """
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    subdir = pathlib.Path.home() / klayout_folder / klayout_subdir_name
    dest = subdir / package_name
    subdir.mkdir(exist_ok=True, parents=True)
    make_link(src, dest)


def install_klayout_package() -> None:
    """Install gdsfactory klayout package.

    Equivalent to using klayout package manager.

    """
    cwd = pathlib.Path(__file__).resolve().parent
    _install_to_klayout(
        src=cwd / "klayout", klayout_subdir_name="salt", package_name="gdsfactory"
    )


def install_klayout_technology(tech_dir: pathlib.Path, tech_name: Optional[str] = None):
    """Install technology to KLayout."""
    _install_to_klayout(
        src=tech_dir,
        klayout_subdir_name="tech",
        package_name=tech_name or tech_dir.name,
    )


if __name__ == "__main__":
    cwd = pathlib.Path(__file__).resolve().parent
    home = pathlib.Path.home()
    src = cwd / "klayout" / "tech"

    install_gdsdiff()
    install_klayout_package()
