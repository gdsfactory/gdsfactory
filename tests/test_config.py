from __future__ import annotations

import pytest

from gdsfactory import config


def test_print_version_plugins_handles_plugin_initialization_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def import_plugin(name: str) -> None:
        if name == "broken_plugin":
            raise RuntimeError("native library initialization failed")
        raise ImportError

    monkeypatch.setattr(config.importlib, "import_module", import_plugin)

    config.print_version_plugins(packages=["broken_plugin"])

    output = capsys.readouterr().out
    assert "broken_plugin" in output
    assert "failed to import" in output
    assert "(RuntimeError)" in output
    assert "native library" in output
    assert "initialization failed" in output
