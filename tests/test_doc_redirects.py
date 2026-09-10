from pathlib import Path

from docs.write_legacy_redirects import write_legacy_redirects


def test_write_legacy_redirects(tmp_path: Path) -> None:
    page = tmp_path / "notebooks" / "03_layer_stack"
    page.mkdir(parents=True)
    (page / "index.html").write_text("canonical page")
    (tmp_path / "index.html").write_text("home page")

    redirects = write_legacy_redirects(tmp_path)

    redirect = tmp_path / "notebooks" / "03_layer_stack.html"
    assert redirects == [redirect]
    assert 'content="0; url=03_layer_stack/"' in redirect.read_text()
    assert "location.search + location.hash" in redirect.read_text()
    assert not tmp_path.with_suffix(".html").exists()
