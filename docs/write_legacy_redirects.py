"""Generate redirects from legacy ``.html`` documentation URLs."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


def write_legacy_redirects(site_dir: Path) -> list[Path]:
    """Create ``page.html`` redirects for directory-style documentation pages."""
    redirects = []
    for index_file in site_dir.rglob("index.html"):
        if index_file.parent == site_dir:
            continue

        redirect_file = index_file.parent.with_suffix(".html")
        target = f"{index_file.parent.name}/"
        escaped_target = html.escape(target, quote=True)
        redirect_file.write_text(
            "<!doctype html>\n"
            '<html lang="en">\n'
            "<head>\n"
            f'  <meta http-equiv="refresh" content="0; url={escaped_target}">\n'
            f'  <link rel="canonical" href="{escaped_target}">\n'
            "  <script>\n"
            f"    location.replace({target!r} + location.search + location.hash);\n"
            "  </script>\n"
            "</head>\n"
            f'<body><a href="{escaped_target}">Continue to the documentation</a>.</body>\n'
            "</html>\n"
        )
        redirects.append(redirect_file)
    return redirects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", type=Path)
    args = parser.parse_args()
    write_legacy_redirects(args.site_dir)
