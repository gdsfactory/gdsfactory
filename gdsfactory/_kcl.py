from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import kfactory as kf


@contextmanager
def temporary_kcl(name: str) -> Iterator[kf.KCLayout]:
    """Yield a temporary KCLayout and unregister it on every exit path."""
    kcl = kf.KCLayout(name=name)
    had_error = False
    try:
        yield kcl
    except BaseException:
        had_error = True
        raise
    finally:
        try:
            if kcl.library is not None:
                kcl.library.delete()
        except Exception:
            if not had_error:
                raise
        finally:
            if kf.layout.kcls.get(kcl.name) is kcl:
                del kf.layout.kcls[kcl.name]
