import warnings

from gdsfactory.config import __next_major_version__


def deprecate(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead. "
        f"{old_name} will be removed in {__next_major_version__}.",
        DeprecationWarning,
        2,
    )
