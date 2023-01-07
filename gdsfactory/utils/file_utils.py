import pathlib
from typing import Union


def append_file_extension(
    filename: Union[str, pathlib.Path], extension: str
) -> Union[str, pathlib.Path]:
    """Try appending extension to file."""
    # Handle whether given with '.'
    if "." not in extension:
        extension = f".{extension}"

    if isinstance(filename, str) and not filename.endswith(extension):
        filename += extension

    if isinstance(filename, pathlib.Path) and not str(filename).endswith(extension):
        filename = filename.with_suffix(extension)
    return filename
