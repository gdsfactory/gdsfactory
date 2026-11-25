"""Publish a built documentation site as one version of the versioned docs.

The versioned site follows the mike layout read by the theme's version selector:
every version lives in its own directory, aliases (e.g. ``stable``) are copies of
a version, and ``versions.json`` at the site root lists them all. The root
``index.html`` redirects to the default alias and ``404.html`` sends pre-versioning
links (``/gdsfactory/notebooks/...``) to the same page under the default alias.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_ALIAS = "stable"


def _version_key(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def load_versions(site_dir: Path) -> list[dict[str, object]]:
    """Return the entries of ``versions.json``, or an empty list."""
    versions_file = site_dir / "versions.json"
    if not versions_file.exists():
        return []
    return json.loads(versions_file.read_text())


def publish_version(
    build_dir: Path,
    site_dir: Path,
    version: str,
    aliases: list[str],
    title: str | None = None,
) -> list[dict[str, object]]:
    """Copy ``build_dir`` into ``site_dir/version`` and register it."""
    site_dir.mkdir(parents=True, exist_ok=True)
    versions = load_versions(site_dir)

    for entry in versions:
        entry["aliases"] = [a for a in entry["aliases"] if a not in aliases]  # type: ignore[union-attr]
    versions = [entry for entry in versions if entry["version"] != version]
    versions.append({"version": version, "title": title or version, "aliases": aliases})

    named = [e for e in versions if _version_key(str(e["version"]))]
    unnamed = [e for e in versions if not _version_key(str(e["version"]))]
    named.sort(key=lambda e: _version_key(str(e["version"])), reverse=True)
    versions = unnamed + named

    for name in [version, *aliases]:
        target = site_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build_dir, target)

    (site_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (site_dir / ".nojekyll").touch()
    write_root_pages(site_dir, versions)
    return versions


def write_root_pages(site_dir: Path, versions: list[dict[str, object]]) -> None:
    """Write the root redirect and the legacy-link 404 page."""
    has_default = any(DEFAULT_ALIAS in e["aliases"] for e in versions)  # type: ignore[operator]
    default = DEFAULT_ALIAS if has_default else str(versions[0]["version"])
    known = sorted(
        {str(e["version"]) for e in versions}
        | {str(a) for e in versions for a in e["aliases"]}  # type: ignore[union-attr]
    )

    (site_dir / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        f'  <meta http-equiv="refresh" content="0; url={default}/">\n'
        f'  <link rel="canonical" href="{default}/">\n'
        "  <script>\n"
        f"    location.replace({default + '/'!r} + location.search + location.hash);\n"
        "  </script>\n"
        "</head>\n"
        f'<body><a href="{default}/">Continue to the documentation</a>.</body>\n'
        "</html>\n"
    )

    # GitHub Pages serves the root 404.html for any missing path of the project
    # site. Old unversioned links get forwarded to the default version.
    (site_dir / "404.html").write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>Page not found</title>\n"
        "  <script>\n"
        "    (function () {\n"
        f"      var versions = {json.dumps(known)};\n"
        f"      var fallback = {default!r};\n"
        '      var parts = location.pathname.split("/");\n'
        "      // parts: ['', 'gdsfactory', <first segment>, ...]\n"
        "      if (parts.length > 2 && versions.indexOf(parts[2]) === -1) {\n"
        "        parts.splice(2, 0, fallback);\n"
        '        location.replace(parts.join("/") + location.search + location.hash);\n'
        "      }\n"
        "    })();\n"
        "  </script>\n"
        "</head>\n"
        "<body>\n"
        f'  <p>Page not found. <a href="/gdsfactory/{default}/">Go to the documentation</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path, help="built HTML of this version")
    parser.add_argument("site_dir", type=Path, help="checkout of the gh-pages branch")
    parser.add_argument("version", help="version directory, e.g. 9.51 or dev")
    parser.add_argument("--alias", action="append", default=[], dest="aliases")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    for entry in publish_version(
        args.build_dir, args.site_dir, args.version, args.aliases, args.title
    ):
        print(entry)
