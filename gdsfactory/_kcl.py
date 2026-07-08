from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import kfactory as kf


@contextmanager
def temporary_kcl(name: str) -> Iterator[kf.KCLayout]:
    """Yield a temporary KCLayout and unregister it on every exit path."""
    kcl = kf.KCLayout(name=name)
    try:
        yield kcl
    finally:
        try:
            kcl.library.delete()
        finally:
            if kf.layout.kcls.get(kcl.name) is kcl:
                del kf.layout.kcls[kcl.name]
