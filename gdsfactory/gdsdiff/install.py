from __future__ import annotations

import configparser
import pathlib
import shutil

if __name__ == "__main__":
    home = pathlib.Path.home()
    config_path = home / ".gitconfig"

    cwd = pathlib.Path.cwd()
    src = cwd / ".gitattributes"
    shutil.copy(src, home)

    config = configparser.RawConfigParser()
    config.read(config_path)
    key = 'diff "gds_diff"'

    if key not in config.sections():
        config.add_section(key)
        config.set(key, "command", "python -m gdsfactory.gdsdiff.gds_diff_git")
        config.set(key, "binary", "True")

        with open(config_path, "w") as f:
            print("installing gds_diff")
            config.write(f, space_around_delimiters=True)
