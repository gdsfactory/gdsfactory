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


def clear_cache(kcl: kf.KCLayout = kf.kcl) -> None:
    """Clears the layout and the cell factory caches for the given layout.

    Removes every cell from the layout and drops the results memoized by each
    registered cell factory, so that the next call rebuilds them.

    Only factories registered on `kcl` are reachable.
    Those decorated with `register_factory=False` keep their cache.
    """
    kcl.clear_kcells()
    for factory in kcl.factories.all():
        factory.cache.clear()
    for virtual_factory in kcl.virtual_factories.all():
        virtual_factory.cache.clear()
