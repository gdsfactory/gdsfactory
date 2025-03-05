import warnings

from gdsfactory.config import __next_major_version__


def deprecate(old_name: str, new_name: str | None = None, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{old_name} is deprecated."
        + (f" Use {new_name} instead." if new_name else "")
        + f" It will be removed in {__next_major_version__}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
