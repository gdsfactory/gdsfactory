"""Deprecation utilities for gdsfactory.

Provides a structured deprecation system with version tracking,
decorator support, and centralized deprecation registry.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from gdsfactory.config import __version__, __next_major_version__

from kfactory import logger

F = TypeVar("F", bound=Callable[..., Any])

# Centralized registry of all active deprecations.
# Each entry maps a feature name to its deprecation metadata.
_DEPRECATION_REGISTRY: dict[str, dict[str, str | None]] = {}


def deprecate(old_name: str, new_name: str | None = None, stacklevel: int = 3) -> None:
    """Issue a deprecation warning for a feature.

    Args:
        old_name: Name of the deprecated feature.
        new_name: Name of the replacement (if any).
        stacklevel: Stack level for the warning.
    """
    warnings.warn(
        f"{old_name} is deprecated."
        + (f" Use {new_name} instead." if new_name else "")
        + f" It will be removed in {__next_major_version__}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(
    *,
    reason: str | None = None,
    replacement: str | None = None,
    since: str | None = None,
    removal: str | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a function or class as deprecated.

    Args:
        reason: Why this is deprecated.
        replacement: What to use instead.
        since: Version when deprecation was introduced.
        removal: Target version for removal (defaults to next major).

    Returns:
        Decorated function that emits a DeprecationWarning on first call.

    Example:
        @deprecated(replacement="new_function", since="9.40.0")
        def old_function():
            ...
    """
    removal_version = removal or __next_major_version__

    def decorator(func: F) -> F:
        name = func.__qualname__

        # Register in the centralized registry
        _DEPRECATION_REGISTRY[name] = {
            "replacement": replacement,
            "since": since,
            "removal": removal_version,
            "reason": reason,
        }

        msg_parts = [f"{name} is deprecated."]
        if reason:
            msg_parts.append(f" Reason: {reason}")
        if replacement:
            msg_parts.append(f" Use {replacement} instead.")
        if since:
            msg_parts.append(f" Deprecated since v{since}.")
        msg_parts.append(f" Will be removed in v{removal_version}.")
        msg = "".join(msg_parts)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def get_deprecation_registry() -> dict[str, dict[str, str | None]]:
    """Return a copy of the current deprecation registry.

    Returns:
        Dictionary mapping feature names to their deprecation metadata.
    """
    return dict(_DEPRECATION_REGISTRY)


def check_deprecations_due(version: str | None = None) -> list[str]:
    """Check for deprecations that are past their removal target.

    Args:
        version: Version to check against (defaults to current version).

    Returns:
        List of feature names that should have been removed.
    """
    check_version = version or __version__
    due = []
    for name, meta in _DEPRECATION_REGISTRY.items():
        removal = meta.get("removal")
        if removal and _version_gte(check_version, removal):
            due.append(name)
    return due


def _version_gte(current: str, target: str) -> bool:
    """Check if current version >= target version (simple tuple comparison)."""
    try:
        current_parts = tuple(int(x) for x in current.split(".")[:3])
        target_parts = tuple(int(x) for x in target.split(".")[:3])
        return current_parts >= target_parts
    except (ValueError, AttributeError):
        logger.debug(
            f"Could not parse versions for comparison: {current!r} vs {target!r}"
        )
        return False
