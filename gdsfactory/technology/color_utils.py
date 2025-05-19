from __future__ import annotations


def ensure_six_digit_hex_color(color: str | int) -> str:
    # Fast path: is already normalized
    if isinstance(color, int):
        return f"#{color:06x}"
    # Remove 0x if present and replace with '#'
    if color.startswith("0x"):
        color = "#" + color[2:]
    # Expand short hex, e.g. #abc -> #aabbcc
    if color.startswith("#") and len(color) == 4:
        color = "#" + color[1] * 2 + color[2] * 2 + color[3] * 2
    return color


_LAYER_COLORS = [
    "#3dcc5c",
    "#2b0fff",
    "#cc3d3d",
    "#e5dd45",
    "#7b3dcc",
    "#cc860c",
    "#73ff0f",
    "#2dccb4",
    "#ff0fa3",
    "#0ec2e6",
    "#3d87cc",
    "#e5520e",
]
