from pp.config import __version__


def equal_or_greater_than(version: str):
    """Raises error if version  is not high enough"""

    if __version__ < version:
        raise ValueError(
            f"gdsfactory minimum required version = {version}\n"
            f"not compatible your current installed version {__version__}\n"
            "you can run:\n"
            "pip install gdsfactory --upgrade\n"
            "to install a later version",
        )


if __name__ == "__main__":
    # equal_or_greater_than("3.0.0")
    # equal_or_greater_than("2.4.9")
    # equal_or_greater_than("2.4.8")
    equal_or_greater_than("2.5.0")
