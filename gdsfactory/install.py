"""Install Klayout and GIT plugins."""

from __future__ import annotations

import configparser
import os
import pathlib
import shutil
import sys

from pygit2 import (
    GIT_CHECKOUT_FORCE,
    GIT_CHECKOUT_RECREATE_MISSING,
    GitError,
    Repository,
    clone_repository,
)

from gdsfactory.config import PATH

home = pathlib.Path.home()


def remove_path_or_dir(dest: pathlib.Path) -> None:
    if dest.is_symlink():
        os.unlink(dest)
    elif dest.is_dir():
        shutil.rmtree(dest)
    else:
        os.remove(dest)


def make_link(src: pathlib.Path, dest: pathlib.Path, overwrite: bool = True) -> None:
    dest = pathlib.Path(dest)
    if dest.exists() and not overwrite:
        print(f"{dest} already exists")
        return
    if dest.exists() or dest.is_symlink():
        print(f"removing {dest} already installed")
        remove_path_or_dir(dest)
    try:
        os.symlink(src, dest, target_is_directory=True)
        print("Symlink made:")
    except OSError as err:
        print("Could not create symlink!")
        print("     Error: ", err)
        if sys.platform == "win32":
            # https://stackoverflow.com/questions/32877260/privlege-error-trying-to-create-symlink-using-python-on-windows-10
            shutil.copytree(src, dest)
            print("Copied directory:")
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


def clone_or_update_repository(url: str, path: str | pathlib.Path) -> None:
    """Clone a repository or update it if it already exists.

    Args:
        url: Git repository URL.
        path: Destination path for the repository.

    This function handles the case where the repository already exists by:
    1. Checking if the path exists and is a git repository
    2. If it exists and is a git repo, fetch the latest changes from the default remote
    3. If it doesn't exist, clone the repository
    4. If it exists but is not a git repo, raise an error
    """
    path = pathlib.Path(path)

    if not path.exists():
        print(f"Cloning {url} to {path}...")
        clone_repository(url, str(path))
        print(f"Successfully cloned {url}")
        return

    try:
        repo = Repository(path)
        print(f"Repository already exists at {path}")

        # Try to fetch and update from remote
        try:
            if repo.remotes:
                remote = repo.remotes[0]
                print(f"Fetching latest changes from {remote.name}...")
                remote.fetch()

                # Get the default branch
                if not repo.head_is_unborn:
                    branch_name = repo.head.shorthand
                    remote_branch = f"{remote.name}/{branch_name}"

                    # Check if remote branch exists and update to it
                    try:
                        repo.references[f"refs/remotes/{remote_branch}"]
                        # Update working directory to match remote (discards local changes)
                        repo.checkout(
                            f"refs/remotes/{remote_branch}",
                            strategy=GIT_CHECKOUT_FORCE | GIT_CHECKOUT_RECREATE_MISSING,
                        )
                        repo.set_head(f"refs/heads/{branch_name}")
                        print(f"Updated to latest from {remote_branch}")
                    except KeyError:
                        print(
                            f"Remote branch '{remote_branch}' not found. "
                            "Continuing with existing repository state"
                        )
                    except GitError as e:
                        print(
                            f"Git operation failed while updating: {e}. "
                            "Continuing with existing repository state"
                        )
            else:
                print("No remotes configured, using existing repository")
        except GitError as e:
            print(
                f"Could not fetch updates from remote: {e}. "
                "Continuing with existing repository"
            )

    except GitError:
        raise ValueError(
            f"'{path}' exists but is not a git repository. "
            f"Please remove it or use a different path."
        )


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

    # install layermap
    _install_to_klayout(
        src=cwd / "gpdk" / "klayout",
        klayout_subdir_name="salt",
        package_name="gdsfactory",
    )

    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    subdir = home / klayout_folder / "salt"

    # install metainfo-ports
    clone_or_update_repository(
        "git@github.com:gdsfactory/metainfo-ports.git", subdir / "metainfo-ports"
    )

    # install klive
    clone_or_update_repository("git@github.com:gdsfactory/klive.git", subdir / "klive")


def install_klayout_technology(
    tech_dir: pathlib.Path, tech_name: str | None = None
) -> None:
    """Install technology to KLayout."""
    _install_to_klayout(
        src=tech_dir,
        klayout_subdir_name="tech",
        package_name=tech_name or tech_dir.name,
    )


py_files = list(PATH.notebooks.glob("**/*.py"))


def convert_py_to_ipynb(
    files: list[pathlib.Path] = py_files,
    output_folder: pathlib.Path = PATH.cwd / "notebooks",
) -> None:
    """Convert notebooks from markdown to ipynb."""
    import jupytext

    output_folder.mkdir(exist_ok=True, parents=True)

    for file in files:
        notebook_file = f"{output_folder}/{file.stem}.ipynb"
        nb = jupytext.read(file)
        jupytext.write(nb, notebook_file)
