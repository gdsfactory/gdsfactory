from typing import ClassVar

import pytest

from gdsfactory import _kcl


class _CleanupError(RuntimeError):
    pass


class _PrimaryError(RuntimeError):
    pass


class _FailingLibrary:
    def delete(self) -> None:
        raise _CleanupError("cleanup failed")


class _SuccessfulLibrary:
    def delete(self) -> None:
        pass


class _FakeLayout:
    kcls: ClassVar[dict[str, "_FakeKcl"]] = {}


class _FakeKcl:
    def __init__(self, name: str) -> None:
        self.name = name
        self.library = _FailingLibrary()
        _FakeLayout.kcls[name] = self


class _FakeKFactory:
    KCLayout = _FakeKcl
    layout = _FakeLayout


def test_temporary_kcl_preserves_body_exception_on_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLayout.kcls.clear()
    monkeypatch.setattr(_kcl, "kf", _FakeKFactory)

    with pytest.raises(_PrimaryError, match="primary failed"):
        with _kcl.temporary_kcl("temporary"):
            raise _PrimaryError("primary failed")

    assert "temporary" not in _FakeLayout.kcls


def test_temporary_kcl_raises_cleanup_error_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLayout.kcls.clear()
    monkeypatch.setattr(_kcl, "kf", _FakeKFactory)

    with pytest.raises(_CleanupError, match="cleanup failed"):
        with _kcl.temporary_kcl("temporary"):
            pass

    assert "temporary" not in _FakeLayout.kcls


def test_temporary_kcl_does_not_unregister_replaced_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLayout.kcls.clear()
    monkeypatch.setattr(_kcl, "kf", _FakeKFactory)
    replacement = object()

    with _kcl.temporary_kcl("temporary") as kcl:
        kcl.library = _SuccessfulLibrary()
        _FakeLayout.kcls[kcl.name] = replacement  # type: ignore[assignment]

    assert _FakeLayout.kcls["temporary"] is replacement
