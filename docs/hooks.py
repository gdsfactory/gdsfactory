"""Post-process notebook markdown: convert MyST admonitions to Material style."""

import re
import sys
from pathlib import Path


def process(markdown: str) -> str:
    """Apply all transformations to a markdown string."""
    markdown = _myst_to_material_admonitions(markdown)
    markdown = _escape_curly_braces(markdown)
    return markdown


def _escape_curly_braces(markdown: str) -> str:
    r"""Escape {WORD} patterns that mkdocstrings would try to resolve.

    Skips content inside code blocks, inline code, and math delimiters.
    """
    _IGNORE = re.compile(
        r"```.*?```|`[^`\n]+`|\$\$.*?\$\$|\$(?!\s)[^$]+?(?<!\s)\$",
        re.DOTALL,
    )
    parts = _IGNORE.split(markdown)
    ignored = _IGNORE.findall(markdown)
    for i, part in enumerate(parts):
        parts[i] = re.sub(r"\{([A-Z_]+)\}", r"\1", part)
    result: list[str] = []
    for i, part in enumerate(parts):
        result.append(part)
        if i < len(ignored):
            result.append(ignored[i])
    return "".join(result)


def _myst_to_material_admonitions(markdown: str) -> str:
    r"""Convert MyST ```{type}\n...\n``` to Material !!! type\n\n    ..."""

    def _replace(match: re.Match[str]) -> str:
        kind = match.group("kind")
        body = match.group("body")
        lines = body.splitlines()
        title = ""
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and (m := re.match(r"^#+ +(.+)", lines[0])):
            title = m.group(1)
            lines.pop(0)
        non_empty = [line for line in lines if line.strip()]
        min_indent = min(
            (len(line) - len(line.lstrip()) for line in non_empty), default=0
        )
        dedented = [line[min_indent:] for line in lines]
        indented = ["    " + line if line.strip() else "" for line in dedented]
        header = f'!!! {kind} "{title}"' if title else f"!!! {kind}"
        return header + "\n\n" + "\n".join(indented)

    return re.sub(
        r"(?ms)^(?P<fence>`{3,})\{(?P<kind>\w+)\}\s*\n(?P<body>.*?)\n(?P=fence)$",
        _replace,
        markdown,
    )


def main() -> None:
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        processed = process(text)
        if processed != text:
            path.write_text(processed, encoding="utf-8")


if __name__ == "__main__":
    main()
